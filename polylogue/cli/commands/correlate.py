"""Session-to-git correlation commands (#1690 phase 1)."""

from __future__ import annotations

import subprocess

import click

from polylogue.cli.shared.types import AppEnv


@click.command("correlate")
@click.argument("conversation_id")
@click.option("--repo-path", "-r", default=None, help="Path to git repository. Defaults to current directory.")
@click.option("--since-hours", "-w", type=int, default=2, help="Hours before/after session to scan for commits.")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def correlate_command(
    env: AppEnv,
    conversation_id: str,
    repo_path: str | None,
    since_hours: int,
    output_format: str | None,
) -> None:
    """Show git commits within the session's time window (#1690 phase 1)."""
    from datetime import timedelta

    from polylogue.api.sync.bridge import run_coroutine_sync

    conv = run_coroutine_sync(env.polylogue.get_conversation(conversation_id))
    if conv is None:
        env.ui.error(f"Conversation not found: {conversation_id}")
        raise SystemExit(1)

    # Determine time window from conversation timestamps.
    start = conv.created_at
    end = conv.updated_at
    if start is None or end is None:
        env.ui.error("Conversation has no timestamp data.")
        raise SystemExit(1)

    window_start = start - timedelta(hours=since_hours)
    window_end = end + timedelta(hours=since_hours)

    # Determine repo path.
    repo = repo_path
    if not repo:
        meta = conv.provider_meta or {}
        repo_url = meta.get("git_repository_url") or conv.git_repository_url
        cwd = meta.get("cwd") or ""
        repo = cwd if cwd and not repo_url else "."

    # Get touched file paths from the session.
    file_paths: set[str] = set()
    for msg in conv.messages:
        if msg.provider_meta:
            paths = msg.provider_meta.get("touched_paths", [])
            if isinstance(paths, list):
                file_paths.update(str(p) for p in paths)

    # Run git log.
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "log",
                "--since",
                window_start.isoformat(),
                "--until",
                window_end.isoformat(),
                "--format=%H%n%ai%n%s%n---",
                "--name-only",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        env.ui.error(f"git log failed: {exc}")
        raise SystemExit(1) from exc

    commits = _parse_git_log(result.stdout, file_paths)

    if output_format == "json":
        import json

        env.ui.console.print(
            json.dumps(
                {
                    "conversation_id": conversation_id,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "repo": str(repo),
                    "commits": commits,
                },
                indent=2,
            )
        )
        return

    env.ui.console.print(f"\n[bold]Session:[/bold] {conversation_id}")
    env.ui.console.print(
        f"[bold]Window:[/bold] {window_start.strftime('%Y-%m-%d %H:%M')} → {window_end.strftime('%Y-%m-%d %H:%M')}"
    )
    env.ui.console.print(f"[bold]Repo:[/bold] {repo}")
    env.ui.console.print(f"[bold]Touched files:[/bold] {len(file_paths)}")
    if not commits:
        env.ui.console.print("\n[dim]No commits found in session window.[/dim]")
        return

    env.ui.console.print(f"\n[bold]Commits in window:[/bold] {len(commits)}")
    for c in commits:
        marker = " [green]●[/green]" if c["overlaps_files"] else "  "
        env.ui.console.print(f"{marker} [bold]{c['short_hash']}[/bold] {c['date'][:10]}  {c['subject'][:72]}")


def _parse_git_log(output: str, session_files: set[str]) -> list[dict[str, object]]:
    """Parse git log output into commit dicts with file-overlap detection."""
    commits: list[dict[str, object]] = []
    blocks = output.split("\n---\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        commit_hash = lines[0].strip()
        date_str = lines[1].strip()
        subject = lines[2].strip()
        changed_files = {f.strip() for f in lines[3:] if f.strip()}
        overlaps = bool(session_files & changed_files) if session_files else False
        commits.append(
            {
                "hash": commit_hash,
                "short_hash": commit_hash[:8],
                "date": date_str,
                "subject": subject,
                "changed_files": sorted(changed_files)[:20],
                "overlaps_files": overlaps,
            }
        )
    return commits
