"""Real-process contracts for the CLI's interactive boundary.

The unit tests for selection and completion cover their component logic.  This
module deliberately crosses the process and PTY boundary so it catches
regressions in TTY detection, external-picker invocation, and Click's shell
completion protocol.
"""

from __future__ import annotations

import os
import shlex
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests.infra.pty_cli import grid_to_text, run_in_pty
from tests.infra.storage_records import DbFactory


def _interactive_env(
    cli_workspace: dict[str, Path],
    tmp_path: Path,
    *,
    picker_dir: Path | None = None,
) -> dict[str, str]:
    """Return a process-isolated environment that keeps the CLI interactive."""
    home = tmp_path / "interactive-home"
    home.mkdir()
    env = {
        "HOME": str(home),
        "XDG_DATA_HOME": str(cli_workspace["data_root"]),
        "XDG_STATE_HOME": str(cli_workspace["state_dir"]),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "POLYLOGUE_ARCHIVE_ROOT": str(cli_workspace["archive_root"]),
        "POLYLOGUE_SITE_CONFIG": "",
        "POLYLOGUE_DAEMON_URL": "http://127.0.0.1:1",
        "POLYLOGUE_SCHEMA_VALIDATION": "off",
    }
    if picker_dir is not None:
        env["PATH"] = f"{picker_dir}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}"
    return env


def _install_first_row_fzf(
    directory: Path,
    trace_path: Path,
    *,
    size_path: Path | None = None,
) -> None:
    """Install a deterministic fzf stand-in that selects its first candidate.

    The CLI still invokes its real external-process picker path.  Returning the
    first input row makes result ordering observable: swapping candidate order
    changes the selected session and fails the route test.
    """
    executable = directory / "fzf"
    terminal_probe = "" if size_path is None else f"stty size < /dev/tty > {shlex.quote(str(size_path))}\n"
    script = "".join(
        (
            "#!/bin/sh\n",
            terminal_probe,
            f"printf '%s\\n' \"$@\" > {shlex.quote(str(trace_path))}\n",
            "IFS= read -r selection || exit 1\n",
            "printf '%s\\n' \"$selection\"\n",
        )
    )
    executable.write_text(script, encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)


def _seed_picker_sessions(cli_workspace: dict[str, Path]) -> None:
    factory = DbFactory(cli_workspace["db_path"])
    now = datetime(2026, 7, 12, tzinfo=timezone.utc)
    for native_id, title, updated_at in (
        ("older", "Interactive picker candidate older", now - timedelta(days=1)),
        ("newer", "Interactive picker candidate newer", now),
    ):
        factory.create_session(
            id=native_id,
            provider="chatgpt",
            title=title,
            messages=[{"id": f"{native_id}-message", "role": "user", "text": "picker route fixture"}],
            created_at=updated_at,
            updated_at=updated_at,
        )


def test_click_bash_completion_protocol_runs_in_a_real_pty(
    cli_workspace: dict[str, Path],
    tmp_path: Path,
) -> None:
    """The installed Click completion protocol emits root options in a PTY."""
    result = run_in_pty(
        [],
        env={
            **_interactive_env(cli_workspace, tmp_path),
            "_POLYLOGUE_COMPLETE": "bash_complete",
            "COMP_WORDS": "polylogue --ori",
            "COMP_CWORD": "1",
        },
    )

    assert result.exit_code == 0
    assert "--origin" in grid_to_text(result.grid)


def test_select_uses_fzf_first_ranked_candidate_in_a_real_pty(
    cli_workspace: dict[str, Path],
    tmp_path: Path,
) -> None:
    """The real CLI hands ranked rows to fzf and prints fzf's chosen session."""
    _seed_picker_sessions(cli_workspace)
    picker_dir = tmp_path / "bin"
    picker_dir.mkdir()
    trace_path = tmp_path / "fzf-options.txt"
    _install_first_row_fzf(picker_dir, trace_path)

    result = run_in_pty(
        ["--sort", "date", "--reverse", "find", "title:Interactive", "then", "select"],
        env=_interactive_env(cli_workspace, tmp_path, picker_dir=picker_dir),
    )

    assert result.exit_code == 0
    assert grid_to_text(result.grid).splitlines()[-1] == "chatgpt-export:ext-older"
    assert trace_path.read_text(encoding="utf-8").splitlines() == [
        "--delimiter",
        "\t",
        "--with-nth",
        "2",
        "--height",
        "40%",
        "--reverse",
        "--preview",
        "echo {} | cut -f3-",
        "--preview-window",
        "down:4:wrap",
    ]


def test_select_propagates_requested_pty_size_to_fzf(
    cli_workspace: dict[str, Path],
    tmp_path: Path,
) -> None:
    """The PTY harness gives the real selector route the requested terminal size."""
    _seed_picker_sessions(cli_workspace)
    picker_dir = tmp_path / "bin"
    picker_dir.mkdir()
    trace_path = tmp_path / "fzf-options.txt"
    size_path = tmp_path / "fzf-terminal-size.txt"
    _install_first_row_fzf(picker_dir, trace_path, size_path=size_path)

    result = run_in_pty(
        ["--sort", "date", "--reverse", "find", "title:Interactive", "then", "select"],
        rows=41,
        cols=137,
        env={
            **_interactive_env(cli_workspace, tmp_path, picker_dir=picker_dir),
            "FZF_DEFAULT_OPTS": "",
        },
    )

    assert result.exit_code == 0
    assert grid_to_text(result.grid).splitlines()[-1] == "chatgpt-export:ext-older"
    # fzf receives its size through the controlling TTY, not through pyte's
    # renderer; the stand-in reads that terminal directly.
    assert size_path.read_text(encoding="utf-8").strip() == "41 137"
