"""``maintenance embedding-orphan-reconcile``: reconcile embeddings.db orphans."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import click

from polylogue.config import Config
from polylogue.paths import archive_file_set_root_for_paths, archive_root, db_path, render_root

if TYPE_CHECKING:
    from polylogue.storage.embeddings.reconcile import EmbeddingOrphanReconcileReport

# Mirrors polylogue.storage.embeddings.reconcile.DEFAULT_QUIET_WINDOW_MS // 1000.
# Hardcoded (not imported) so this decorator's default doesn't force the
# embeddings.reconcile module -- and its own heavy import chain -- onto the
# `--help` path; test_embeddings_defaults_match_reconcile_module asserts these
# stay in sync.
_DEFAULT_QUIET_WINDOW_SECONDS = 300


@click.command("embedding-orphan-reconcile")
@click.option(
    "--max-count",
    type=int,
    default=None,
    help="Maximum number of orphan embedding rows to reconcile or preview (default: unbounded for inspect).",
)
@click.option(
    "--quiet-window-seconds",
    type=int,
    default=_DEFAULT_QUIET_WINDOW_SECONDS,
    show_default=True,
    help="Skip candidates embedded more recently than this window (races an in-flight rebuild).",
)
@click.option(
    "--sample-limit",
    type=int,
    default=30,
    show_default=True,
    help="Maximum number of representative samples to include.",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Delete orphan embedding rows. Without this flag, the command only inspects and reports (break-glass).",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def embedding_orphan_reconcile_command(
    max_count: int | None,
    quiet_window_seconds: int,
    sample_limit: int,
    yes: bool,
    output_format: str,
) -> None:
    """Inspect (default) or reconcile embeddings.db rows orphaned by an index rebuild.

    An index rebuild (full re-ingest, ``ops reset --index``, a provider
    full-replace parse) can leave ``message_embeddings_meta`` /
    ``message_embeddings`` / ``embedding_status`` rows in ``embeddings.db``
    pointing at message/session identities that no longer exist in the
    rebuilt ``index.db``. Daemon convergence reconciles these automatically
    in bounded batches; this command is the manual inspect/break-glass path.
    """
    from polylogue.storage.embeddings.reconcile import DEFAULT_MAX_COUNT, reconcile_embedding_orphans

    root = archive_file_set_root_for_paths(archive_root_path=archive_root(), db_anchor=db_path())
    index_db = root / "index.db"
    embeddings_db = root / "embeddings.db"
    if yes:
        from polylogue.maintenance.offline_guard import running_daemon_pid
        from polylogue.storage.index_generation import RebuildLease, RebuildLeaseUnavailableError

        active_config = Config(archive_root=root, render_root=render_root(), sources=[], db_path=index_db)
        try:
            with RebuildLease(root):
                daemon_pid = running_daemon_pid(active_config)
                if daemon_pid is not None:
                    raise click.ClickException(
                        f"embedding orphan reconcile refused while polylogued PID {daemon_pid} is running"
                    )
                report = reconcile_embedding_orphans(
                    index_db,
                    embeddings_db,
                    dry_run=False,
                    max_count=DEFAULT_MAX_COUNT if max_count is None else max_count,
                    sample_size=sample_limit,
                    quiet_window_ms=quiet_window_seconds * 1000,
                    mutation_authority="offline-exclusive",
                )
        except RebuildLeaseUnavailableError as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        report = reconcile_embedding_orphans(
            index_db,
            embeddings_db,
            dry_run=True,
            max_count=max_count,
            sample_size=sample_limit,
            quiet_window_ms=quiet_window_seconds * 1000,
        )
    payload = {
        "mode": "embedding_orphan_reconcile",
        "mutates": bool(yes),
        **report.to_dict(),
    }

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    _render_embedding_orphan_reconcile_plain(report)


def _render_embedding_orphan_reconcile_plain(report: EmbeddingOrphanReconcileReport) -> None:
    click.echo("Embedding orphan reconcile")
    click.echo(f"Index DB:      {report.index_db}")
    click.echo(f"Embeddings DB: {report.embeddings_db}")
    click.echo(f"Mode:          {'dry-run' if report.dry_run else 'apply'}")
    click.echo(
        f"Scanned:       {report.scanned_message_meta_rows:,} message meta row(s), "
        f"{report.scanned_vector_rows:,} vector row(s), "
        f"{report.scanned_status_rows:,} status row(s)"
    )
    click.echo(
        f"Orphans:       {report.orphan_message_rows:,} message identity row(s) "
        f"({report.orphan_message_meta_rows:,} meta, {report.orphan_vector_rows:,} vector), "
        f"{report.orphan_status_rows:,} status row(s)"
    )
    if report.skipped_recent_message_rows or report.skipped_recent_status_rows:
        click.echo(
            "Quiet-skipped: "
            f"{report.skipped_recent_message_rows:,} message row(s), "
            f"{report.skipped_recent_status_rows:,} status row(s) "
            f"(within {report.quiet_window_ms // 1000}s)"
        )
    if report.dry_run:
        click.echo(
            f"Would remove:  {report.candidate_message_meta_rows:,} meta row(s), "
            f"{report.candidate_vector_rows:,} vector row(s), "
            f"{report.candidate_status_rows:,} status row(s)"
        )
    else:
        click.echo(
            f"Removed:       {report.removed_message_rows:,} meta row(s), "
            f"{report.removed_vector_rows:,} vector row(s), {report.removed_status_rows:,} status row(s)"
        )
    if report.sessions_recounted:
        click.echo(f"Recounted:     {report.sessions_recounted:,} session(s) message_count_embedded")
    click.echo(f"More pending:  {report.more_pending}")
    if report.samples:
        click.echo("Samples:")
        for sample in report.samples[:5]:
            target = sample.message_id or sample.session_id
            click.echo(f"  {sample.action} {sample.kind} {target}")
