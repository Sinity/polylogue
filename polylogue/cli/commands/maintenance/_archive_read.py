"""``maintenance archive-read``: read index sessions from the archive."""

from __future__ import annotations

import json

import click

from polylogue.archive.query.transaction import archive_read_context
from polylogue.paths import archive_file_set_root_for_paths, archive_root, db_path


@click.command("archive-read")
@click.option("--query", "-q", type=str, default=None, help="Search block text instead of listing sessions.")
@click.option("--origin", type=str, default=None, help="Restrict reads to one origin token.")
@click.option("--limit", "-l", type=int, default=20, show_default=True, help="Maximum rows to return.")
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def archive_read_command(query: str | None, origin: str | None, limit: int, output_format: str) -> None:
    """Read index sessions from the archive."""
    root = archive_file_set_root_for_paths(archive_root_path=archive_root(), db_anchor=db_path())
    index_db_path = root / "index.db"
    if not index_db_path.exists():
        message = f"archive index.db does not exist: {index_db_path}"
        if output_format == "json":
            click.echo(json.dumps({"ok": False, "error": message, "index_db_path": str(index_db_path)}, indent=2))
        else:
            click.echo(f"Blocked: {message}", err=True)
        raise SystemExit(1)

    with archive_read_context(
        root,
        operation="cli.maintenance.archive_read",
        arguments={"query": query, "origin": origin, "limit": limit},
        page_size=limit,
        projection=output_format,
        workload_class="scan" if query else "interactive",
    ) as archive:
        if query is not None:
            hits = archive.search_summaries(query, limit=limit, origin=origin)
            if output_format == "json":
                click.echo(
                    json.dumps(
                        {
                            "ok": True,
                            "mode": "search",
                            "query": query,
                            "origin": origin,
                            "index_db_path": str(index_db_path),
                            "hits": [
                                {
                                    "rank": hit.rank,
                                    "session_id": hit.session_id,
                                    "block_id": hit.block_id,
                                    "message_id": hit.message_id,
                                    "origin": hit.origin,
                                    "title": hit.title,
                                    "snippet": hit.snippet,
                                }
                                for hit in hits
                            ],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return
            for hit in hits:
                title = f" {hit.title}" if hit.title else ""
                click.echo(f"{hit.rank}. {hit.session_id}{title}")
                click.echo(f"   {hit.block_id}: {hit.snippet}")
            return

        summaries = archive.list_summaries(limit=limit, origin=origin)
        if output_format == "json":
            click.echo(
                json.dumps(
                    {
                        "ok": True,
                        "mode": "list",
                        "origin": origin,
                        "index_db_path": str(index_db_path),
                        "sessions": [
                            {
                                "session_id": summary.session_id,
                                "native_id": summary.native_id,
                                "origin": summary.origin,
                                "title": summary.title,
                                "created_at": summary.created_at,
                                "updated_at": summary.updated_at,
                                "message_count": summary.message_count,
                                "word_count": summary.word_count,
                                "tags": list(summary.tags),
                            }
                            for summary in summaries
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        for summary in summaries:
            title = f" {summary.title}" if summary.title else ""
            click.echo(f"{summary.session_id}{title}")
            click.echo(f"  {summary.origin} messages={summary.message_count:,} words={summary.word_count:,}")
