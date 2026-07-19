"""``maintenance embeddings-rescue``: copy vectors from a retired embeddings tier.

Break-glass, offline-only migration path for polylogue-04kl: a retired
``embeddings.db.v2-retired-YYYYMMDD`` tier can hold hundreds of thousands of
Voyage vectors whose ``message_id`` + ``content_hash`` identity still matches
the freshly rebuilt index. ``--plan`` (default) is a read-only census;
``--yes`` copies vectors for every *fully rescuable* session (see
:mod:`polylogue.storage.embeddings.rescue` for why rescue is session-, not
message-, granular) directly into the live ``embeddings.db``, skipping a live
Voyage API re-embed for those sessions entirely.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import click

from polylogue.config import Config
from polylogue.paths import archive_file_set_root_for_paths, archive_root, db_path, render_root

if TYPE_CHECKING:
    from polylogue.storage.embeddings.rescue import (
        EmbeddingRescueExecuteReport,
        EmbeddingRescuePlanReport,
    )


@click.command("embeddings-rescue")
@click.option(
    "--source",
    "source_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the retired embeddings.db tier to rescue vectors from.",
)
@click.option(
    "--plan",
    "plan_only",
    is_flag=True,
    default=True,
    show_default=True,
    help="Read-only census of rescuable vectors (default). Mutually exclusive with --yes.",
)
@click.option(
    "--yes",
    "apply",
    is_flag=True,
    help="Copy vectors for every fully-rescuable session into the live embeddings.db.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Maximum number of sessions to rescue in this invocation (apply mode only; resumable).",
)
@click.option(
    "--sample-limit",
    type=int,
    default=30,
    show_default=True,
    help="Maximum number of representative non-matched samples to include.",
)
@click.option(
    "--sample-verify-count",
    type=int,
    default=20,
    show_default=True,
    help="Number of rescued vectors to sample-verify byte-identical against the source (apply mode only).",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def embeddings_rescue_command(
    source_path: str,
    plan_only: bool,
    apply: bool,
    limit: int | None,
    sample_limit: int,
    sample_verify_count: int,
    output_format: str,
) -> None:
    """Inspect (default) or apply a content-hash vector rescue from a retired embeddings tier."""
    del plan_only  # --plan is documentation-only; --yes/apply is the sole control switch.
    root = archive_file_set_root_for_paths(archive_root_path=archive_root(), db_anchor=db_path())
    index_db = root / "index.db"

    if not apply:
        from polylogue.storage.embeddings.rescue import plan_embedding_rescue

        plan_report = plan_embedding_rescue(index_db, source_path, sample_size=sample_limit)
        payload = {"mutates": False, **plan_report.to_dict()}
        if output_format == "json":
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
            return
        _render_plan_plain(plan_report)
        return

    from polylogue.maintenance.offline_guard import running_daemon_pid
    from polylogue.storage.embeddings.rescue import execute_embedding_rescue

    active_config = Config(archive_root=root, render_root=render_root(), sources=[], db_path=index_db)
    daemon_pid = running_daemon_pid(active_config)
    if daemon_pid is not None:
        raise click.ClickException(f"embeddings rescue refused while polylogued PID {daemon_pid} is running")

    exec_report = execute_embedding_rescue(
        index_db,
        source_path,
        limit=limit,
        sample_size=sample_limit,
        sample_verify_count=sample_verify_count,
        mutation_authority="offline-exclusive",
    )
    payload = {"mutates": True, **exec_report.to_dict()}
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    _render_execute_plain(exec_report)


def _render_plan_plain(report: EmbeddingRescuePlanReport) -> None:
    click.echo("Embeddings rescue plan (read-only)")
    click.echo(f"Index DB:      {report.index_db}")
    click.echo(f"Source DB:     {report.source_db}")
    click.echo(f"Model:         {report.model}")
    click.echo(f"Eligible sessions:        {report.counts.eligible_sessions:,}")
    click.echo(
        f"Fully rescuable sessions: {report.counts.fully_rescuable_sessions:,} "
        f"({report.counts.rescuable_messages:,} message(s))"
    )
    click.echo(
        f"Partial sessions:         {report.counts.partial_sessions:,} "
        f"({report.counts.partial_matched_messages:,} matched message(s), not written -- see command help)"
    )
    click.echo(
        "Skipped messages:         "
        f"missing={report.counts.skipped_missing:,} "
        f"hash_mismatch={report.counts.skipped_hash_mismatch:,} "
        f"model_mismatch={report.counts.skipped_model_mismatch:,}"
    )
    if report.samples:
        click.echo("Samples:")
        for sample in report.samples[:5]:
            click.echo(f"  {sample.status} {sample.message_id} (session {sample.session_id})")


def _render_execute_plain(report: EmbeddingRescueExecuteReport) -> None:
    click.echo("Embeddings rescue apply")
    click.echo(f"Index DB:       {report.index_db}")
    click.echo(f"Source DB:      {report.source_db}")
    click.echo(f"Embeddings DB:  {report.embeddings_db}")
    click.echo(f"Model:          {report.model}")
    click.echo(f"Rescued:        {report.rescued_sessions:,} session(s), {report.rescued_messages:,} message(s)")
    click.echo(
        "Skipped:        "
        f"already_fresh={report.skipped_already_fresh_sessions:,} "
        f"race={report.skipped_race_sessions:,} "
        f"missing={report.counts.skipped_missing:,} "
        f"hash_mismatch={report.counts.skipped_hash_mismatch:,} "
        f"model_mismatch={report.counts.skipped_model_mismatch:,}"
    )
    click.echo(f"Sample verify:  {report.verified_sample_ok:,}/{report.verified_sample_total:,} ok")
    click.echo(f"More pending:   {report.more_pending}")
    click.echo(f"Status:         {'OK' if report.ok else 'VERIFICATION FAILED'}")
