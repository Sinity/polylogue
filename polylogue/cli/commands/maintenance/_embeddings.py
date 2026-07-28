"""``maintenance embedding-orphan-reconcile``: inspect embeddings.db orphans.

Read-only by design (automagic-invariants, polylogue-gd6v/4jsk): daemon
convergence (``periodic_embedding_orphan_reconcile_check``,
``polylogue/daemon/embedding_backlog.py``) already reconciles this backlog
automatically in bounded batches, so a manual mutate/apply path here would
be exactly the redundant "demoted to break-glass" surface that ruling
forbids rather than deletes. This command stays as pure diagnostic preview.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import click

from polylogue.paths import archive_root
from polylogue.storage.archive_identity import ArchiveLocation

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
    help="Maximum number of orphan embedding rows to preview (default: unbounded).",
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
    output_format: str,
) -> None:
    """Inspect embeddings.db rows orphaned by an index rebuild. Read-only.

    An index rebuild (full re-ingest, ``ops reset --index``, a provider
    full-replace parse) can leave ``message_embeddings_meta`` /
    ``message_embeddings`` / ``embedding_status`` rows in ``embeddings.db``
    pointing at message/session identities that no longer exist in the
    rebuilt ``index.db``. Daemon convergence reconciles these automatically
    in bounded batches; this command is diagnostic preview only.
    """
    from polylogue.storage.embeddings.reconcile import reconcile_embedding_orphans

    location = ArchiveLocation.resolve(archive_root())
    index_db = location.active_index_path
    embeddings_db = location.configured_root / "embeddings.db"
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
        "mutates": False,
        **report.to_dict(),
    }

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    _render_embedding_orphan_reconcile_plain(report)


def _render_embedding_orphan_reconcile_plain(report: EmbeddingOrphanReconcileReport) -> None:
    """Render the reconcile report honestly under the v4 content-addressed model.

    polylogue-q88p: ``message_embeddings``/``message_embeddings_meta`` are
    content-addressed and shared -- this reconciler never deletes them (a
    vector's underlying hash may still be referenced by another live
    message). Only ``message_embedding_refs`` (the per-message mapping) is
    ever removed, so the report speaks in terms of refs, not "meta"/"vector
    row(s) removed" -- that wording would claim a mutation that never
    happens. ``scanned_message_meta_rows``/``scanned_vector_rows`` remain
    informational counts of the (deduped) content-addressed tables.
    """
    click.echo("Embedding orphan reconcile (inspect, read-only)")
    click.echo(f"Index DB:      {report.index_db}")
    click.echo(f"Embeddings DB: {report.embeddings_db}")
    click.echo(
        f"Scanned:       {report.scanned_message_meta_rows:,} distinct vector meta row(s) "
        f"(content-addressed, shared), {report.scanned_vector_rows:,} distinct vector row(s), "
        f"{report.scanned_status_rows:,} status row(s)"
    )
    click.echo(
        f"Orphans:       {report.orphan_message_rows:,} orphan message ref(s), "
        f"{report.orphan_status_rows:,} status row(s)"
    )
    if report.skipped_recent_message_rows or report.skipped_recent_status_rows:
        click.echo(
            "Quiet-skipped: "
            f"{report.skipped_recent_message_rows:,} message row(s), "
            f"{report.skipped_recent_status_rows:,} status row(s) "
            f"(within {report.quiet_window_ms // 1000}s)"
        )
    click.echo(
        f"Would remove:  {report.candidate_message_rows:,} message ref(s), "
        f"{report.candidate_status_rows:,} status row(s)"
    )
    if report.sessions_recounted:
        click.echo(f"Recounted:     {report.sessions_recounted:,} session(s) message_count_embedded")
    click.echo(f"More pending:  {report.more_pending}")
    if report.samples:
        click.echo("Samples:")
        for sample in report.samples[:5]:
            target = sample.message_id or sample.session_id
            click.echo(f"  {sample.action} {sample.kind} {target}")
