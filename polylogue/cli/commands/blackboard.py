"""polylogue blackboard — persistent agent-addressable notes (#1697)."""

from __future__ import annotations

import click

from polylogue.cli.shared.types import AppEnv


@click.group("blackboard")
def blackboard_command() -> None:
    """Persistent agent-addressable notes surface."""


@blackboard_command.command("post")
@click.option(
    "--kind",
    "-k",
    required=True,
    type=click.Choice(["finding", "blocker", "decision", "handoff", "question", "observation"]),
)
@click.option("--title", "-t", required=True)
@click.option("--content", "-c", required=True)
@click.option("--scope-repo")
@click.option("--scope-session")
@click.option("--scope-issue", type=int)
@click.option("--scope-path")
@click.option("--related-sessions", "-r", multiple=True)
@click.pass_obj
def blackboard_post(
    env: AppEnv,
    kind: str,
    title: str,
    content: str,
    scope_repo: str | None,
    scope_session: str | None,
    scope_issue: int | None,
    scope_path: str | None,
    related_sessions: tuple[str, ...],
) -> None:
    """Post a note to the blackboard."""
    import json
    import uuid
    from datetime import datetime, timezone

    note_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    related_json = json.dumps(list(related_sessions))

    from polylogue.paths import db_path

    db = db_path()
    if not db.exists():
        env.ui.error("No archive database found. Run polylogued first.")
        raise SystemExit(1)

    import sqlite3

    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            """INSERT INTO blackboard_notes
               (note_id, kind, title, content, scope_repo, scope_session,
                scope_issue, scope_path, related_session_ids_json,
                created_at, materialized_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (note_id, kind, title, content, scope_repo, scope_session, scope_issue, scope_path, related_json, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    env.ui.console.print(f"[bold green]Posted {kind} note:[/bold green] {note_id}")


@blackboard_command.command("list")
@click.option(
    "--kind", "-k", type=click.Choice(["finding", "blocker", "decision", "handoff", "question", "observation"])
)
@click.option("--scope-repo", "-r")
@click.option("--unresolved", is_flag=True, help="Only show unresolved notes (blockers, questions).")
@click.option("--limit", "-l", type=int, default=20)
@click.pass_obj
def blackboard_list(
    env: AppEnv,
    kind: str | None,
    scope_repo: str | None,
    unresolved: bool,
    limit: int,
) -> None:
    """List blackboard notes."""
    from polylogue.paths import db_path

    db = db_path()
    if not db.exists():
        env.ui.error("No archive database found.")
        raise SystemExit(1)

    import sqlite3

    conn = sqlite3.connect(str(db))
    try:
        if not _table_exists(conn, "blackboard_notes"):
            env.ui.console.print("[dim]No blackboard notes yet.[/dim]")
            return

        where = ["1=1"]
        params: list[object] = []
        if kind:
            where.append("kind = ?")
            params.append(kind)
        if scope_repo:
            where.append("scope_repo = ?")
            params.append(scope_repo)
        if unresolved:
            where.append("resolved_at IS NULL")

        rows = conn.execute(
            f"SELECT note_id, kind, title, content, scope_repo, scope_session, "
            f"scope_issue, scope_path, created_at, resolved_at "
            f"FROM blackboard_notes WHERE {' AND '.join(where)} "
            f"ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()

        if not rows:
            env.ui.console.print("[dim]No matching notes.[/dim]")
            return

        for row in rows:
            status = "[green]open[/green]" if row[9] is None else "[dim]resolved[/dim]"
            env.ui.console.print(
                f"[bold]{row[1]}[/bold] {status}  {row[2]}\n"
                f"  {row[3][:120]}{'...' if len(row[3] or '') > 120 else ''}\n"
                f"  id={row[0]}  repo={row[4] or '-'}  created={row[8]}\n"
            )
    finally:
        conn.close()


def _table_exists(conn: object, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,)).fetchone()  # type: ignore[attr-defined]
    return row is not None
