"""``polylogue user-state blackboard`` — persistent agent-addressable notes (#1697)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.paths import archive_file_set_root_for_paths, archive_root, db_path
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import (
    ArchiveBlackboardNoteEnvelope,
    list_archive_blackboard_note_envelopes,
    upsert_blackboard_note,
)

_KIND_CHOICES = ("finding", "blocker", "decision", "handoff", "question", "observation")
_KIND_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")


class _ParsedBlackboardBody(TypedDict):
    kind: str
    title: str
    content: str
    scope_repo: str | None


@click.group("blackboard")
def blackboard_command() -> None:
    """Persistent agent-addressable notes surface."""


@blackboard_command.command("post")
@click.option(
    "--kind",
    "-k",
    required=True,
    type=click.Choice(_KIND_CHOICES),
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
    import sqlite3
    import uuid

    note_id = str(uuid.uuid4())
    user_db = _user_db_path()
    initialize_archive_database(user_db, ArchiveTier.USER)
    body = _blackboard_body(
        kind=kind,
        title=title,
        content=content,
        scope_repo=scope_repo,
        scope_issue=scope_issue,
        scope_path=scope_path,
        related_sessions=related_sessions,
    )
    target_type = "session" if scope_session else None

    conn = sqlite3.connect(user_db)
    try:
        upsert_blackboard_note(
            conn,
            body,
            target_type=target_type,
            target_id=scope_session,
            note_id=note_id,
        )
        conn.commit()
    finally:
        conn.close()
    env.ui.console.print(f"[bold green]Posted {kind} note:[/bold green] {note_id}")


@blackboard_command.command("list")
@click.option("--kind", "-k", type=click.Choice(_KIND_CHOICES))
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
    import sqlite3

    user_db = _user_db_path()
    if not user_db.exists():
        env.ui.console.print("[dim]No blackboard notes yet.[/dim]")
        return

    conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
    try:
        notes = list_archive_blackboard_note_envelopes(conn)
        if not notes:
            env.ui.console.print("[dim]No blackboard notes yet.[/dim]")
            return

        filtered: list[tuple[ArchiveBlackboardNoteEnvelope, _ParsedBlackboardBody]] = []
        for note in notes:
            parsed = _parse_blackboard_body(note.body)
            if kind and parsed["kind"] != kind:
                continue
            if scope_repo and parsed["scope_repo"] != scope_repo:
                continue
            if unresolved and parsed["kind"] not in {"blocker", "question"}:
                continue
            if len(filtered) >= limit:
                break
            filtered.append((note, parsed))

        if not filtered:
            env.ui.console.print("[dim]No matching notes.[/dim]")
            return

        for note, parsed in filtered:
            status = "[green]open[/green]"
            content = parsed["content"]
            env.ui.console.print(
                f"[bold]{parsed['kind']}[/bold] {status}  {parsed['title']}\n"
                f"  {content[:120]}{'...' if len(content) > 120 else ''}\n"
                f"  id={note.note_id}  repo={parsed['scope_repo'] or '-'}  created_ms={note.created_at_ms}\n"
            )
    finally:
        conn.close()


def _user_db_path() -> Path:
    return archive_file_set_root_for_paths(archive_root_path=archive_root(), db_anchor=db_path()) / "user.db"


def _blackboard_body(
    *,
    kind: str,
    title: str,
    content: str,
    scope_repo: str | None,
    scope_issue: int | None,
    scope_path: str | None,
    related_sessions: tuple[str, ...],
) -> str:
    lines = [f"[{kind}] {title}".strip(), "", content]
    scope_lines = []
    if scope_repo:
        scope_lines.append(f"scope_repo: {scope_repo}")
    if scope_issue is not None:
        scope_lines.append(f"scope_issue: {scope_issue}")
    if scope_path:
        scope_lines.append(f"scope_path: {scope_path}")
    if related_sessions:
        scope_lines.append(f"related_sessions: {', '.join(related_sessions)}")
    if scope_lines:
        lines.extend(["", *scope_lines])
    return "\n".join(lines)


def _parse_blackboard_body(body: str) -> _ParsedBlackboardBody:
    first, _, rest = body.partition("\n")
    match = _KIND_RE.match(first)
    kind = match.group(1) if match else "observation"
    title = match.group(2) if match else first
    content_lines: list[str] = []
    scope_repo: str | None = None
    for line in rest.strip().splitlines():
        if line.startswith("scope_repo: "):
            scope_repo = line.removeprefix("scope_repo: ").strip()
            continue
        if line.startswith(("scope_issue: ", "scope_path: ", "related_sessions: ")):
            continue
        content_lines.append(line)
    content = "\n".join(content_lines).strip()
    return {"kind": kind, "title": title, "content": content, "scope_repo": scope_repo}
