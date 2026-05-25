"""Embedding activation, preflight, backfill, disable, and status commands.

This is the operator-facing onboarding surface for the embedding pipeline
described in [`docs/architecture.md`](../../../docs/architecture.md) —
``polylogue embed enable`` (alias: ``activate``) writes the config flip and
records the Voyage API key in ``polylogue.toml``; ``polylogue embed preflight``
counts the conversations that would be embedded plus the Voyage cost estimate
without contacting the provider; ``polylogue embed backfill`` runs the first
batch with per-conversation cost feedback against the cost cap; and
``polylogue embed disable`` flips the gate back off without dropping any
existing embeddings.

The substrate-side primitives (token-count and cost estimation,
``PendingConversation`` enumeration, ``embed_conversation_sync``) live under
``polylogue.storage.embeddings``; the CLI is a thin orchestrator over them.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import click

from polylogue.cli.shared.embed_stats import show_embedding_stats
from polylogue.cli.shared.types import AppEnv

# Resolution order for the API key:
#   1. explicit --voyage-api-key flag on the activate command
#   2. existing voyage_api_key in user TOML (set by a prior activation)
#   3. VOYAGE_API_KEY environment variable
# Only #1 and #3 are accepted by ``activate``; #2 is reused on the second
# activation so existing keys are not lost.


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Cost preflight numbers for the configured archive."""

    total_conversations: int
    pending_conversations: int
    pending_messages: int
    estimated_tokens: int
    estimated_cost_usd: float
    model: str
    dimension: int
    cost_cap_usd: float
    windowed: bool = False
    max_conversations: int | None = None
    max_messages: int | None = None


def _read_pending_message_count(
    db_path: Path,
    *,
    rebuild: bool = False,
    max_conversations: int | None = None,
    max_messages: int | None = None,
) -> tuple[int, int, int]:
    """Return ``(total_convs, pending_convs, pending_messages)``.

    Pending = no ``embedding_status`` row, or ``needs_reindex = 1``.
    Reading happens against a sync read connection so the command works even
    when the daemon is not running.
    """
    from polylogue.storage.sqlite.connection import open_read_connection

    with open_read_connection(db_path) as conn:
        total = int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0])
        if max_conversations is not None or max_messages is not None:
            from polylogue.api import select_pending_embedding_conversation_window

            pending = select_pending_embedding_conversation_window(
                conn,
                rebuild=rebuild,
                max_conversations=max_conversations,
                max_messages=max_messages,
            )
            return total, len(pending), sum(item.message_count for item in pending)
        try:
            if rebuild:
                pending_convs = total
                pending_messages = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
                return total, pending_convs, pending_messages
            pending_convs = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM conversations c
                    LEFT JOIN embedding_status e
                      ON c.conversation_id = e.conversation_id
                    WHERE e.conversation_id IS NULL OR e.needs_reindex = 1
                    """
                ).fetchone()[0]
            )
            pending_messages = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM messages m
                    JOIN conversations c ON c.conversation_id = m.conversation_id
                    LEFT JOIN embedding_status e
                      ON c.conversation_id = e.conversation_id
                    WHERE e.conversation_id IS NULL OR e.needs_reindex = 1
                    """
                ).fetchone()[0]
            )
        except sqlite3.OperationalError:
            # embedding_status table may not exist on a fresh / never-embedded DB.
            pending_convs = total
            pending_messages = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
    return total, pending_convs, pending_messages


def _build_preflight_report(
    env: AppEnv,
    *,
    rebuild: bool = False,
    max_conversations: int | None = None,
    max_messages: int | None = None,
) -> PreflightReport:
    """Build a :class:`PreflightReport` without contacting Voyage."""
    from polylogue.config import load_polylogue_config
    from polylogue.storage.search_providers.sqlite_vec_support import (
        ESTIMATED_TOKENS_PER_MESSAGE,
        VOYAGE_4_COST_PER_1M_TOKENS,
    )

    cfg = load_polylogue_config()
    total, pending, pending_messages = _read_pending_message_count(
        env.config.db_path,
        rebuild=rebuild,
        max_conversations=max_conversations,
        max_messages=max_messages,
    )
    estimated_tokens = pending_messages * ESTIMATED_TOKENS_PER_MESSAGE
    estimated_cost = estimated_tokens * VOYAGE_4_COST_PER_1M_TOKENS / 1_000_000
    return PreflightReport(
        total_conversations=total,
        pending_conversations=pending,
        pending_messages=pending_messages,
        estimated_tokens=estimated_tokens,
        estimated_cost_usd=estimated_cost,
        model=cfg.embedding_model,
        dimension=cfg.embedding_dimension,
        cost_cap_usd=cfg.embedding_max_cost_usd,
        windowed=max_conversations is not None or max_messages is not None,
        max_conversations=max_conversations,
        max_messages=max_messages,
    )


def _render_preflight(env: AppEnv, report: PreflightReport) -> None:
    console = env.ui.console
    console.print("\n[bold]Embedding preflight[/bold]")
    console.print(f"  Model:                 {report.model} ({report.dimension}d)")
    console.print(f"  Total conversations:   {report.total_conversations:,}")
    if report.windowed:
        limits: list[str] = []
        if report.max_conversations is not None:
            limits.append(f"{report.max_conversations:,} conversations")
        if report.max_messages is not None:
            limits.append(f"{report.max_messages:,} messages")
        console.print(f"  Window limit:          {', '.join(limits)}")
        console.print(f"  Window conversations:  {report.pending_conversations:,}")
        console.print(f"  Window messages:       {report.pending_messages:,}")
    else:
        console.print(f"  Pending conversations: {report.pending_conversations:,}")
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


# ---------------------------------------------------------------------------
# TOML write helper — preserves existing file contents and inserts/updates the
# ``[embedding]`` section in-place. Uses simple line manipulation rather than
# a full TOML rewrite so user comments and unrelated sections survive.
# ---------------------------------------------------------------------------


def _embedding_section_lines(*, enabled: bool, voyage_api_key: str | None) -> list[str]:
    body: list[str] = ["[embedding]", f"enabled = {str(enabled).lower()}"]
    if voyage_api_key:
        body.append(f'voyage_api_key = "{voyage_api_key}"')
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
    """Return the user TOML path the activate flow should write to.

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
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    updated = _splice_embedding_section(existing, enabled=enabled, voyage_api_key=voyage_api_key)
    path.write_text(updated, encoding="utf-8")
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
    click.echo("Run [bold]polylogue embed backfill[/bold] to start the first embedding batch, or restart polylogued.")


@embed_command.command("activate")
@click.pass_context
def activate_subcommand(ctx: click.Context) -> None:
    """Alias for ``embed enable`` retained as a discoverable verb."""
    ctx.forward(enable_subcommand)


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
    click.echo("Existing embeddings remain; re-run [bold]polylogue embed enable[/bold] to resume.")


@embed_command.command("preflight")
@click.option(
    "--max-conversations",
    type=click.IntRange(min=1),
    default=None,
    help="Estimate only the next bounded conversation window.",
)
@click.option(
    "--max-messages",
    type=click.IntRange(min=1),
    default=None,
    help="Estimate only the next bounded message window.",
)
@click.option(
    "--rebuild",
    is_flag=True,
    help="Estimate re-embedding every conversation, not just pending ones.",
)
@click.pass_obj
def preflight_subcommand(env: AppEnv, max_conversations: int | None, max_messages: int | None, rebuild: bool) -> None:
    """Estimate token count and Voyage cost for the pending backlog.

    Read-only — does not touch the embedding provider. Use this before
    ``embed enable`` or ``embed backfill`` to budget the first run.
    """
    report = _build_preflight_report(
        env,
        rebuild=rebuild,
        max_conversations=max_conversations,
        max_messages=max_messages,
    )
    _render_preflight(env, report)


@embed_command.command("backfill")
@click.option(
    "--max-conversations",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum conversations to process in this catch-up window.",
)
@click.option(
    "--max-messages",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum messages to include in this catch-up window.",
)
@click.option(
    "--stop-after-seconds",
    type=click.IntRange(min=1),
    default=None,
    help="Stop starting new conversations after this many seconds.",
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
    help="Re-embed every conversation, not just the pending ones.",
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
    max_conversations: int | None,
    max_messages: int | None,
    stop_after_seconds: int | None,
    max_errors: int | None,
    rebuild: bool,
    yes: bool,
) -> None:
    """Run the first embedding batch with per-conversation cost feedback.

    Prints the cost preflight, confirms, then iterates pending conversations
    (or all conversations if ``--rebuild``) and emits running totals so the
    user can interrupt before the soft monthly cap kicks in.
    """
    from polylogue.storage.embeddings.materialization import (
        embed_conversation_sync,
        iter_pending_conversations,
        mark_all_conversations_needs_reindex,
    )
    from polylogue.storage.embeddings.progress import (
        CatchupRunDelta,
        CatchupRunStart,
        finish_embedding_catchup_run,
        record_embedding_catchup_progress,
        start_embedding_catchup_run,
    )
    from polylogue.storage.search_providers import create_vector_provider
    from polylogue.storage.search_providers.sqlite_vec_support import (
        ESTIMATED_TOKENS_PER_MESSAGE,
        VOYAGE_4_COST_PER_1M_TOKENS,
    )

    key = _resolve_voyage_key(env, None)
    if not key:
        click.echo(
            "Error: Voyage API key not configured. Run [bold]polylogue embed enable[/bold] first.",
            err=True,
        )
        raise click.Abort()

    report = _build_preflight_report(
        env,
        rebuild=rebuild,
        max_conversations=max_conversations,
        max_messages=max_messages,
    )
    _render_preflight(env, report)
    if not yes and not click.confirm("\nProceed with backfill?", default=False):
        click.echo("Cancelled.")
        return
    if rebuild:
        mark_all_conversations_needs_reindex(env.repository.backend)

    vec_provider = create_vector_provider(
        voyage_api_key=key,
        db_path=env.config.db_path,
        model=report.model,
        dimension=report.dimension,
    )
    if vec_provider is None:
        click.echo("Error: vector provider unavailable (sqlite-vec or voyage init failed).", err=True)
        raise click.Abort()

    pending = iter_pending_conversations(
        env.repository.backend,
        rebuild=rebuild,
        max_conversations=max_conversations,
        max_messages=max_messages,
    )
    if not pending:
        click.echo("All conversations are already embedded.")
        return

    cap = report.cost_cap_usd
    cumulative_cost = 0.0
    embedded = 0
    errors = 0
    console = env.ui.console
    started_at = time.monotonic()
    stopped_reason: str | None = None
    run_id = start_embedding_catchup_run(
        env.config.db_path,
        CatchupRunStart(
            rebuild=rebuild,
            max_conversations=max_conversations,
            max_messages=max_messages,
            stop_after_seconds=stop_after_seconds,
            max_errors=max_errors,
            planned_conversations=len(pending),
            planned_messages=sum(item.message_count for item in pending),
        ),
    )

    try:
        for index, item in enumerate(pending, start=1):
            if stop_after_seconds is not None and time.monotonic() - started_at >= stop_after_seconds:
                stopped_reason = f"time limit reached ({stop_after_seconds}s)"
                break
            outcome = embed_conversation_sync(env.repository, vec_provider, item.conversation_id)
            if outcome.status == "embedded":
                embedded += 1
                batch_cost = (
                    outcome.embedded_message_count
                    * ESTIMATED_TOKENS_PER_MESSAGE
                    * VOYAGE_4_COST_PER_1M_TOKENS
                    / 1_000_000
                )
                cumulative_cost += batch_cost
                record_embedding_catchup_progress(
                    env.config.db_path,
                    run_id,
                    CatchupRunDelta(
                        conversation_id=item.conversation_id,
                        embedded=True,
                        embedded_messages=outcome.embedded_message_count,
                        estimated_cost_usd=batch_cost,
                    ),
                )
                console.print(
                    f"  [{index}/{len(pending)}] {item.title or item.conversation_id[:12]}: "
                    f"{outcome.embedded_message_count} msgs (~${batch_cost:.4f}, cumulative ~${cumulative_cost:.4f})"
                )
                if cap > 0 and cumulative_cost > cap:
                    stopped_reason = f"cost cap reached (~${cumulative_cost:.4f} > ${cap:.2f})"
                    console.print(
                        f"[yellow]Cost cap reached (~${cumulative_cost:.4f} > ${cap:.2f}). "
                        f"Stopping after {embedded} conversations.[/yellow]"
                    )
                    break
            elif outcome.status in {"no_messages", "no_embeddable_messages"}:
                record_embedding_catchup_progress(
                    env.config.db_path,
                    run_id,
                    CatchupRunDelta(conversation_id=item.conversation_id, skipped=True),
                )
                console.print(
                    f"  [{index}/{len(pending)}] {item.title or item.conversation_id[:12]}: no embeddable messages"
                )
            elif outcome.status == "error":
                errors += 1
                record_embedding_catchup_progress(
                    env.config.db_path,
                    run_id,
                    CatchupRunDelta(conversation_id=item.conversation_id, errored=True),
                )
                console.print(f"  [{index}/{len(pending)}] {item.conversation_id}: error {outcome.error}")
                if max_errors is not None and errors >= max_errors:
                    stopped_reason = f"max errors reached ({max_errors})"
                    break
    except KeyboardInterrupt:
        finish_embedding_catchup_run(env.config.db_path, run_id, status="interrupted", stop_reason="keyboard interrupt")
        raise
    except Exception as exc:
        finish_embedding_catchup_run(env.config.db_path, run_id, status="failed", stop_reason=str(exc))
        raise

    if stopped_reason:
        finish_embedding_catchup_run(env.config.db_path, run_id, status="stopped", stop_reason=stopped_reason)
    else:
        finish_embedding_catchup_run(env.config.db_path, run_id, status="completed", stop_reason=None)

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
    "activate_subcommand",
    "backfill_subcommand",
    "disable_subcommand",
    "embed_command",
    "enable_subcommand",
    "preflight_subcommand",
    "status_subcommand",
]
