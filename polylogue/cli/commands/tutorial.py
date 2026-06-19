"""Interactive first-run walkthrough — ``polylogue ops tutorial``.

The tutorial is an idempotent five-stage onboarding flow. Re-running picks
up where the previous run left off: each stage probes the archive/config
state and either marks itself ``[skip]`` (already satisfied) or guides the
operator through the next action. ``--non-interactive`` suppresses prompts
and prints the diagnostic block once — used by tests and as a "what would
this do" preview. (#1263)
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from polylogue.cli.shared.types import AppEnv


@dataclass(frozen=True, slots=True)
class TutorialStage:
    """One step in the onboarding flow.

    ``probe`` returns ``(satisfied, message)`` where ``satisfied=True``
    skips the stage and ``message`` is the line we print either way.
    """

    number: int
    title: str
    probe: Callable[[], tuple[bool, str]]
    action_text: str


def _stage_detect_sources() -> tuple[bool, str]:
    """Detect installed chat-source directories."""
    from polylogue.cli.commands.init import detect_chat_sources

    detected = detect_chat_sources()
    present = [d for d in detected if d.present]
    if present:
        names = ", ".join(d.family for d in present)
        return True, f"Found {len(present)} source(s): {names}"
    return False, "No chat-source directories detected yet."


def _stage_starter_config() -> tuple[bool, str]:
    """Verify the starter config has been written."""
    from polylogue.cli.commands.init import starter_config_path

    config_path = starter_config_path()
    if config_path.exists():
        return True, f"Config exists at {config_path}."
    return False, f"No config at {config_path} yet."


def _stage_start_daemon() -> tuple[bool, str]:
    """Probe daemon liveness using the first-run diagnostic."""
    from polylogue.cli.commands.status_diagnostics import diagnose_first_run

    diag = diagnose_first_run(daemon_alive=_daemon_http_alive())
    if diag.kind == "healthy":
        return True, "Daemon is running and archive is healthy."
    if diag.kind == "no_daemon":
        return False, "Daemon is not running."
    # Schema / locked / stale-pid / no_sources fall through here so the
    # tutorial does not falsely claim the daemon is up.
    return False, diag.headline


def _stage_first_search() -> tuple[bool, str]:
    """Report whether the archive has any sessions to search."""
    from polylogue.paths import archive_root, db_path

    db = _active_archive_db(db_path(), archive_root())
    if db is None:
        return False, "No archive yet — ingest must run before search."
    try:
        import sqlite3

        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=0.5)
        try:
            count = _count_searchable_sessions(conn)
        finally:
            conn.close()
    except Exception:
        return False, "Archive present but could not be queried."
    if count > 0:
        return True, f"Archive has {count:,} sessions."
    return False, "Archive present but empty — wait for first ingest to finish."


def _active_archive_db(_db_anchor: Path, root: Path) -> Path | None:
    """Return the archive DB file that contains searchable sessions."""
    archive_db = root / "index.db"
    if archive_db.exists():
        return archive_db
    return None


def _table_exists(conn: Any, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _count_searchable_sessions(conn: Any) -> int:
    if _table_exists(conn, "sessions"):
        count_row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return int(count_row[0]) if count_row else 0
    return 0


def _stage_open_reader() -> tuple[bool, str]:
    """Final stage — nothing to verify, just a pointer."""
    return False, "Reader has not been launched in this session."


STAGES: tuple[TutorialStage, ...] = (
    TutorialStage(
        number=1,
        title="Detect chat sources",
        probe=_stage_detect_sources,
        action_text="Run `polylogue init` to scan and record source roots.",
    ),
    TutorialStage(
        number=2,
        title="Write starter config",
        probe=_stage_starter_config,
        action_text="Run `polylogue init` (writes `polylogue.toml`).",
    ),
    TutorialStage(
        number=3,
        title="Start daemon",
        probe=_stage_start_daemon,
        action_text="Run `polylogued run` in another terminal, then press enter.",
    ),
    TutorialStage(
        number=4,
        title="Your first search",
        probe=_stage_first_search,
        action_text="Try: polylogue 'hello'",
    ),
    TutorialStage(
        number=5,
        title="Open reader",
        probe=_stage_open_reader,
        action_text="Run `polylogue ops dashboard` to open the local reader.",
    ),
)


def _daemon_http_alive() -> bool:
    """Best-effort daemon liveness probe with a short timeout."""
    import os
    from urllib.request import Request, urlopen

    url = os.environ.get("POLYLOGUE_DAEMON_URL", "http://127.0.0.1:8766")
    try:
        req = Request(f"{url}/api/status", method="GET")
        with urlopen(req, timeout=0.5) as resp:
            resp.read(1)
        return True
    except Exception:
        return False


@click.command("tutorial")
@click.option(
    "--non-interactive",
    is_flag=True,
    help="Skip prompts; print the stage diagnostic and exit.",
)
@click.pass_obj
def tutorial_command(env: AppEnv, non_interactive: bool) -> None:
    """Walk through a fresh polylogue install end-to-end.

    Each stage is idempotent — already-satisfied stages are marked
    ``[skip]``. Use ``--non-interactive`` to print the current state
    without prompting (suitable for scripts and CI).
    """
    console = env.ui.console
    total = len(STAGES)
    console.print("\n[bold]polylogue ops tutorial[/bold]")
    console.print("  Five stages will walk you through a working install.\n")

    for stage in STAGES:
        satisfied, message = stage.probe()
        prefix = "[green]✓[/green]" if satisfied else "[yellow]·[/yellow]"
        label = "[skip]" if satisfied else "[do  ]"
        console.print(f"  {prefix} Step {stage.number}/{total} {label} {stage.title}")
        console.print(f"      {message}")
        if not satisfied:
            console.print(f"      → {stage.action_text}")
            if non_interactive:
                # Print the rest of the stages as previews and exit.
                continue
            try:
                click.prompt(
                    "      Press enter when done (or Ctrl-C to exit)",
                    default="",
                    show_default=False,
                    prompt_suffix="",
                )
            except (click.exceptions.Abort, EOFError, KeyboardInterrupt):
                console.print("\n  [dim]Tutorial paused. Re-run `polylogue ops tutorial` to resume.[/dim]")
                sys.exit(0)
    console.print("\n[green]Done.[/green] See `polylogue --help` for more.")


__all__ = ["TutorialStage", "STAGES", "tutorial_command"]
