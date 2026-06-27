"""Embedding activation, preflight, backfill, disable, and status commands.

This is the operator-facing onboarding surface for the embedding pipeline
described in [`docs/architecture.md`](../../../docs/architecture.md) —
``polylogue ops embed enable`` writes the config flip and records the Voyage API
key in ``polylogue.toml``; ``polylogue ops embed preflight`` counts the sessions
that would be embedded plus the Voyage cost estimate without contacting the
provider; ``polylogue ops embed backfill`` runs the first batch with per-session
cost feedback against the cost cap; and
``polylogue ops embed disable`` flips the gate back off without dropping any
existing embeddings.

The substrate-side primitives (token-count and cost estimation,
``PendingSession`` enumeration, ``embed_session_sync``) live under
``polylogue.storage.embeddings``; the CLI is a thin orchestrator over them.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import cast

import click

from polylogue.cli.shared.embed_stats import show_embedding_stats
from polylogue.cli.shared.types import AppEnv
from polylogue.storage.embeddings.preflight import (
    PreflightReport,
    build_preflight_report,
    effective_cost_cap,
    preflight_backfill_args,
    preflight_payload,
)

# Resolution order for the API key:
#   1. explicit --voyage-api-key flag on the enable command
#   2. existing voyage_api_key in user TOML (set by a prior activation)
#   3. VOYAGE_API_KEY environment variable
# Only #1 and #3 are accepted by ``enable``; #2 is reused on the second
# enable run so existing keys are not lost.


def _effective_cost_cap(config_cap_usd: float, run_cap_usd: float | None) -> float:
    return effective_cost_cap(config_cap_usd, run_cap_usd)


def _build_preflight_report(
    env: AppEnv,
    *,
    rebuild: bool = False,
    max_sessions: int | None = None,
    max_messages: int | None = None,
    max_cost_usd: float | None = None,
) -> PreflightReport:
    """Build a :class:`PreflightReport` without contacting Voyage."""
    return build_preflight_report(
        env.config.db_path,
        rebuild=rebuild,
        max_sessions=max_sessions,
        max_messages=max_messages,
        max_cost_usd=max_cost_usd,
    )


def _active_archive_index_path(db_path: Path) -> Path | None:
    from polylogue.paths import archive_root

    candidates = []
    if db_path.name == "index.db":
        candidates.append(db_path)
    candidates.append(db_path.with_name("index.db"))
    candidates.append(archive_root() / "index.db")
    index_db = next((candidate for candidate in dict.fromkeys(candidates) if candidate.exists()), None)
    if index_db is None:
        return None
    try:
        conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions' LIMIT 1").fetchone()
            return index_db if row is not None else None
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def _render_preflight(env: AppEnv, report: PreflightReport) -> None:
    console = env.ui.console
    console.print("\n[bold]Embedding preflight[/bold]")
    console.print(f"  Model:                 {report.model} ({report.dimension}d)")
    console.print(f"  Total sessions:   {report.total_sessions:,}")
    if report.windowed:
        limits: list[str] = []
        if report.max_sessions is not None:
            limits.append(f"{report.max_sessions:,} sessions")
        if report.max_messages is not None:
            limits.append(f"{report.max_messages:,} messages")
        if report.max_cost_usd is not None:
            limits.append(f"${report.max_cost_usd:.4f}")
        console.print(f"  Window limit:          {', '.join(limits)}")
        console.print(f"  Window sessions:  {report.pending_sessions:,}")
        console.print(f"  Window messages:       {report.pending_messages:,}")
    else:
        console.print(f"  Pending sessions: {report.pending_sessions:,}")
        console.print(f"  Pending messages:      {report.pending_messages:,}")
    console.print(f"  Estimated tokens:      ~{report.estimated_tokens:,}")
    console.print(f"  Estimated cost (USD):  ~${report.estimated_cost_usd:.4f}")
    if report.cost_cap_usd > 0:
        console.print(f"  Monthly cost cap:      ${report.cost_cap_usd:.2f} (embedding_max_cost_usd)")
    else:
        console.print("  Monthly cost cap:      [yellow]unbounded[/yellow] (embedding_max_cost_usd=0.0)")
    console.print(
        "\n  [dim]Estimate uses Voyage's $0.10 / 1M tokens at "
        "~500 tokens/message. Voyage does not return token counts, so "
        "the figure is approximate.[/dim]"
    )


def _preflight_backfill_args(report: PreflightReport) -> list[str] | None:
    return preflight_backfill_args(report)


def _preflight_payload(report: PreflightReport) -> dict[str, object]:
    return preflight_payload(report)


def _render_preflight_json(report: PreflightReport) -> None:
    click.echo(json.dumps(_preflight_payload(report), indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# TOML write helper — preserves existing file contents and inserts/updates the
# ``[embedding]`` section in-place. Uses simple line manipulation rather than
# a full TOML rewrite so user comments and unrelated sections survive.
# ---------------------------------------------------------------------------


def _embedding_section_lines(*, enabled: bool, voyage_api_key: str | None) -> list[str]:
    import tomli_w

    table: dict[str, object] = {"enabled": enabled}
    if voyage_api_key:
        table["voyage_api_key"] = voyage_api_key
    # ``tomli_w`` escapes quotes, backslashes, and newlines so a key value
    # containing TOML metacharacters cannot corrupt the persisted file.
    rendered = tomli_w.dumps(table).rstrip("\n")
    body: list[str] = ["[embedding]"]
    if rendered:
        body.extend(rendered.splitlines())
    body.append("")
    return body


def _splice_embedding_section(
    existing: str,
    *,
    enabled: bool,
    voyage_api_key: str | None,
) -> str:
    """Replace or insert the ``[embedding]`` section in ``existing``.

    The section starts at the ``[embedding]`` header and ends at the next
    bracketed section header or end-of-file. When the section is absent the
    new block is appended (with a leading blank line if needed).
    """
    lines = existing.splitlines()
    new_section = _embedding_section_lines(enabled=enabled, voyage_api_key=voyage_api_key)

    start = -1
    for idx, line in enumerate(lines):
        if line.strip() == "[embedding]":
            start = idx
            break

    if start < 0:
        prefix = lines + ([""] if lines and lines[-1].strip() else [])
        return "\n".join(prefix + new_section).rstrip() + "\n"

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("[") and lines[idx].rstrip().endswith("]"):
            end = idx
            break

    rebuilt = lines[:start] + new_section + lines[end:]
    return "\n".join(rebuilt).rstrip() + "\n"


def _resolve_user_config_path() -> Path:
    """Return the user TOML path the enable flow should write to.

    Honors ``POLYLOGUE_CONFIG`` so tests and per-project setups can redirect
    the write; otherwise falls back to the XDG starter path.
    """
    override = os.environ.get("POLYLOGUE_CONFIG")
    if override:
        return Path(override)
    from polylogue.cli.commands.init import starter_config_path

    return starter_config_path()


def _write_embedding_section(*, enabled: bool, voyage_api_key: str | None) -> Path:
    """Persist the embedding configuration to the user TOML."""
    path = _resolve_user_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        updated = _splice_embedding_section(existing, enabled=enabled, voyage_api_key=voyage_api_key)
        path.write_text(updated, encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(
            f"Cannot write embedding config to {path}: {exc.strerror or exc}.\n"
            "This config is read-only — on a Nix/Home-Manager-managed setup it is a\n"
            "store symlink. To activate embeddings, either:\n"
            "  - point POLYLOGUE_CONFIG at a writable polylogue.toml and re-run, or\n"
            "  - set [embedding] enabled = true in your managed config, or\n"
            "  - skip the config write entirely: `polylogue ops embed backfill` reads\n"
            "    VOYAGE_API_KEY from the environment and does not require enable."
        ) from exc
    return path


# ---------------------------------------------------------------------------
# Click group + subcommands
# ---------------------------------------------------------------------------


@click.group("embed")
def embed_command() -> None:
    """Manage the embedding pipeline (activation, preflight, backfill)."""


def _check_sqlite_vec_available() -> tuple[bool, str | None]:
    import importlib.util

    if importlib.util.find_spec("sqlite_vec") is None:
        return False, (
            "sqlite-vec is not installed. Install with:\n"
            "  pip install sqlite-vec\n"
            "or rely on the project devshell which bundles it."
        )
    return True, None


def _resolve_voyage_key(env: AppEnv, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if env.config.index_config and env.config.index_config.voyage_api_key:
        return env.config.index_config.voyage_api_key
    return os.environ.get("VOYAGE_API_KEY") or None


@embed_command.command("enable")
@click.option(
    "--voyage-api-key",
    "voyage_api_key",
    envvar="VOYAGE_API_KEY",
    help="Voyage AI API key. Falls back to VOYAGE_API_KEY env var.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Skip the interactive confirmation prompt.",
)
@click.option(
    "--no-store-key",
    is_flag=True,
    help=(
        "Flip embedding_enabled on but do not record the API key in polylogue.toml; the daemon will continue to read VOYAGE_API_KEY from the environment instead."
    ),
)
@click.pass_obj
def enable_subcommand(
    env: AppEnv,
    voyage_api_key: str | None,
    yes: bool,
    no_store_key: bool,
) -> None:
    """Turn on the embedding pipeline.

    Verifies sqlite-vec is importable, captures the Voyage API key, prints a
    cost preflight, and on confirmation writes ``[embedding]`` into the user
    ``polylogue.toml``. The daemon picks the change up on its next config
    reload.
    """
    available, hint = _check_sqlite_vec_available()
    if not available:
        click.echo(f"Error: {hint}", err=True)
        raise click.Abort()

    key = _resolve_voyage_key(env, voyage_api_key)
    if not key:
        click.echo(
            "Error: Voyage API key not provided. Pass --voyage-api-key or set VOYAGE_API_KEY.",
            err=True,
        )
        raise click.Abort()

    report = _build_preflight_report(env)
    _render_preflight(env, report)

    if not yes and not click.confirm("\nEnable embedding pipeline with the above estimates?", default=False):
        click.echo("Cancelled. No changes made.")
        return

    path = _write_embedding_section(enabled=True, voyage_api_key=None if no_store_key else key)
    click.echo(f"\nEmbeddings enabled. Wrote {path}")
    if no_store_key:
        click.echo("API key not stored in config; ensure VOYAGE_API_KEY remains set for daemon and CLI.")
    click.echo(
        "Run [bold]polylogue ops embed backfill[/bold] to start the first embedding batch, or restart polylogued."
    )


@embed_command.command("disable")
@click.pass_obj
def disable_subcommand(env: AppEnv) -> None:
    """Disable the embedding pipeline without dropping existing embeddings.

    Flips ``embedding.enabled = false`` in the user TOML and keeps the stored
    API key in place. Previously-embedded messages remain queryable via
    ``--similar``; only future ingest stops triggering embedding work.
    """
    cfg_key = env.config.index_config.voyage_api_key if env.config.index_config else None
    path = _write_embedding_section(enabled=False, voyage_api_key=cfg_key)
    click.echo(f"Embeddings disabled. Wrote {path}")
    click.echo("Existing embeddings remain; re-run [bold]polylogue ops embed enable[/bold] to resume.")


@embed_command.command("preflight")
@click.option(
    "--max-sessions",
    type=click.IntRange(min=1),
    default=None,
    help="Estimate only the next bounded session window.",
)
@click.option(
    "--max-messages",
    type=click.IntRange(min=1),
    default=None,
    help="Estimate only the next bounded message window.",
)
@click.option(
    "--max-cost-usd",
    type=click.FloatRange(min=0.0, min_open=True),
    default=None,
    help="Estimate only the next approximate cost window.",
)
@click.option(
    "--rebuild",
    is_flag=True,
    help="Estimate re-embedding every session, not just pending ones.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format.",
)
@click.pass_obj
def preflight_subcommand(
    env: AppEnv,
    max_sessions: int | None,
    max_messages: int | None,
    max_cost_usd: float | None,
    rebuild: bool,
    output_format: str,
) -> None:
    """Estimate token count and Voyage cost for the pending backlog.

    Read-only — does not touch the embedding provider. Use this before
    ``embed enable`` or ``embed backfill`` to budget the first run.
    """
    report = _build_preflight_report(
        env,
        rebuild=rebuild,
        max_sessions=max_sessions,
        max_messages=max_messages,
        max_cost_usd=max_cost_usd,
    )
    if output_format == "json":
        _render_preflight_json(report)
    else:
        _render_preflight(env, report)


@embed_command.command("backfill")
@click.option(
    "--max-sessions",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum sessions to process in this catch-up window.",
)
@click.option(
    "--max-messages",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum messages to include in this catch-up window.",
)
@click.option(
    "--max-cost-usd",
    type=click.FloatRange(min=0.0, min_open=True),
    default=None,
    help="Approximate maximum Voyage cost to include in this catch-up window.",
)
@click.option(
    "--stop-after-seconds",
    type=click.IntRange(min=1),
    default=None,
    help="Stop starting new sessions after this many seconds.",
)
@click.option(
    "--max-errors",
    type=click.IntRange(min=1),
    default=None,
    help="Stop the catch-up window after this many provider errors.",
)
@click.option(
    "--rebuild",
    is_flag=True,
    help="Re-embed every session, not just the pending ones.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Skip the cost confirmation prompt.",
)
@click.pass_obj
def backfill_subcommand(
    env: AppEnv,
    max_sessions: int | None,
    max_messages: int | None,
    max_cost_usd: float | None,
    stop_after_seconds: int | None,
    max_errors: int | None,
    rebuild: bool,
    yes: bool,
) -> None:
    """Run the first embedding batch with per-session cost feedback.

    Prints the cost preflight, confirms, then iterates pending sessions
    (or all sessions if ``--rebuild``) and emits running totals so the
    user can interrupt before the soft monthly cap kicks in.
    """
    from polylogue.storage.search_providers import create_vector_provider

    key = _resolve_voyage_key(env, None)
    if not key:
        click.echo(
            "Error: Voyage API key not configured. Run [bold]polylogue ops embed enable[/bold] first.",
            err=True,
        )
        raise click.Abort()

    report = _build_preflight_report(
        env,
        rebuild=rebuild,
        max_sessions=max_sessions,
        max_messages=max_messages,
        max_cost_usd=max_cost_usd,
    )
    _render_preflight(env, report)
    if not yes and not click.confirm("\nProceed with backfill?", default=False):
        click.echo("Cancelled.")
        return
    index_db = _active_archive_index_path(env.config.db_path)
    if index_db is None:
        click.echo(
            "Error: index.db not found. Initialize the archive tiers first.",
            err=True,
        )
        raise click.Abort()

    if rebuild:
        from polylogue.storage.embeddings.materialization import mark_all_archive_sessions_needs_reindex

        mark_all_archive_sessions_needs_reindex(index_db)

    embeddings_db = index_db.with_name("embeddings.db")
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    vec_provider = create_vector_provider(
        voyage_api_key=key,
        db_path=embeddings_db,
        model=report.model,
        dimension=report.dimension,
    )
    if vec_provider is None:
        click.echo("Error: vector provider unavailable (sqlite-vec or voyage init failed).", err=True)
        raise click.Abort()

    _run_archive_backfill(
        env,
        index_db,
        vec_provider,
        report,
        rebuild=rebuild,
        max_sessions=max_sessions,
        stop_after_seconds=stop_after_seconds,
        max_errors=max_errors,
    )


def _run_archive_backfill(
    env: AppEnv,
    index_db: Path,
    vec_provider: object,
    report: PreflightReport,
    *,
    rebuild: bool,
    max_sessions: int | None,
    stop_after_seconds: int | None,
    max_errors: int | None,
) -> None:
    from polylogue.protocols import VectorProvider
    from polylogue.storage.embeddings.materialization import (
        embed_archive_session_sync,
        select_pending_archive_session_window,
    )
    from polylogue.storage.search_providers.sqlite_vec_support import (
        ESTIMATED_TOKENS_PER_MESSAGE,
        VOYAGE_4_COST_PER_1M_TOKENS,
    )
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    embeddings_db = index_db.with_name("embeddings.db")
    conn = open_readonly_connection(index_db)
    try:
        if embeddings_db.exists():
            conn.execute("ATTACH DATABASE ? AS embeddings", (str(embeddings_db),))
            status_table = "embeddings.embedding_status"
        else:
            status_table = ""
        pending = select_pending_archive_session_window(
            conn,
            status_table=status_table,
            rebuild=rebuild,
            max_sessions=max_sessions,
            max_messages=report.max_messages,
        )
    finally:
        conn.close()
    if not pending:
        click.echo("All sessions are already embedded.")
        return

    cap = _effective_cost_cap(report.cost_cap_usd, report.max_cost_usd)
    cumulative_cost = 0.0
    embedded = 0
    errors = 0
    console = env.ui.console
    started_at = time.monotonic()
    stopped_reason: str | None = None
    typed_provider = cast(VectorProvider, vec_provider)

    for index, item in enumerate(pending, start=1):
        if stop_after_seconds is not None and time.monotonic() - started_at >= stop_after_seconds:
            stopped_reason = f"time limit reached ({stop_after_seconds}s)"
            break
        outcome = embed_archive_session_sync(index_db, typed_provider, item.session_id)
        if outcome.status == "embedded":
            embedded += 1
            batch_cost = (
                outcome.embedded_message_count * ESTIMATED_TOKENS_PER_MESSAGE * VOYAGE_4_COST_PER_1M_TOKENS / 1_000_000
            )
            cumulative_cost += batch_cost
            console.print(
                f"  [{index}/{len(pending)}] {item.title or item.session_id[:12]}: "
                f"{outcome.embedded_message_count} msgs (~${batch_cost:.4f}, cumulative ~${cumulative_cost:.4f})"
            )
            if cap > 0 and cumulative_cost > cap:
                stopped_reason = f"cost cap reached (~${cumulative_cost:.4f} > ${cap:.2f})"
                console.print(
                    f"[yellow]Cost cap reached (~${cumulative_cost:.4f} > ${cap:.2f}). "
                    f"Stopping after {embedded} sessions.[/yellow]"
                )
                break
        elif outcome.status in {"no_messages", "no_embeddable_messages"}:
            console.print(f"  [{index}/{len(pending)}] {item.title or item.session_id[:12]}: no embeddable messages")
        elif outcome.status == "error":
            errors += 1
            console.print(f"  [{index}/{len(pending)}] {item.session_id}: error {outcome.error}")
            if max_errors is not None and errors >= max_errors:
                stopped_reason = f"max errors reached ({max_errors})"
                break

    click.echo(f"\nBackfill complete. Embedded {embedded}, errors {errors}, est. cost ~${cumulative_cost:.4f}.")
    if stopped_reason:
        click.echo(f"Stopped early: {stopped_reason}.")


@embed_command.command("status")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
)
@click.option(
    "--detail",
    is_flag=True,
    help="Include exact pending-message, freshness, model, and retrieval-band accounting.",
)
@click.pass_obj
def status_subcommand(env: AppEnv, output_format: str, detail: bool) -> None:
    """Show embedding coverage and freshness."""
    show_embedding_stats(env, json_output=(output_format == "json"), detail=detail)


__all__ = [
    "PreflightReport",
    "backfill_subcommand",
    "disable_subcommand",
    "embed_command",
    "enable_subcommand",
    "preflight_subcommand",
    "status_subcommand",
]
