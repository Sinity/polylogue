"""Virtual PTY harness for deterministic CLI output capture.

Runs polylogue CLI commands in a pseudo-terminal to capture ANSI escape
sequences, color codes, and cursor movements that plain subprocess
capture would miss. Uses pyte for terminal emulation.

Key capabilities:
- Run CLI commands in a real PTY with controlled dimensions
- Capture raw ANSI output stream
- Render terminal state to a grid (rows × cols) via pyte
- Strip non-deterministic content (timestamps, paths, durations)
- Produce snapshot-friendly text output
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Conditional pyte import with fallback
try:
    import pyte

    HAS_PYTE = True
except ImportError:
    HAS_PYTE = False


@dataclass
class PtyResult:
    """Result from running CLI in a PTY."""

    exit_code: int
    raw_output: bytes
    grid: list[str]
    duration: float

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    def to_text(self, *, strip_trailing: bool = True) -> str:
        """Convert grid to text representation."""
        return grid_to_text(self.grid, strip_trailing=strip_trailing)


def run_in_pty(
    args: list[str],
    *,
    rows: int = 24,
    cols: int = 80,
    env: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> PtyResult:
    """Run polylogue CLI in a pseudo-terminal and capture output.

    Args:
        args: CLI arguments (e.g., ["--help"])
        rows: Terminal height in lines (default 24)
        cols: Terminal width in columns (default 80)
        env: Environment variables to set (merged with clean env)
        timeout: Maximum execution time in seconds

    Returns:
        PtyResult with exit code, raw bytes, rendered grid, and duration

    Raises:
        ImportError: if pyte is not available
        TimeoutError: if command exceeds timeout
    """
    if not HAS_PYTE:
        raise ImportError(
            "pyte is required for PTY testing. "
            "Install with: pip install pyte"
        )

    # Build environment (same clean setup as cli_subprocess.py)
    clean_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "UV_SYSTEM_PYTHON": "1",
        "VOYAGE_API_KEY": "",
        "HOME": os.environ.get("HOME", "/tmp"),
        "TERM": "xterm-256color",  # Set TERM for PTY
    }

    # Preserve mutmut markers
    for key in ("MUTANT_UNDER_TEST", "PY_IGNORE_IMPORTMISMATCH"):
        value = os.environ.get(key)
        if value is not None:
            clean_env[key] = value

    # Merge custom environment
    if env:
        clean_env.update(env)

    # Build command
    project_root = Path(__file__).parent.parent.parent
    command = ["uv", "run", "--project", str(project_root), "polylogue"] + args

    # Handle mutmut case
    if "MUTANT_UNDER_TEST" in clean_env:
        project_root_literal = repr(str(project_root))
        command = [
            sys.executable,
            "-c",
            (
                "import os, runpy, sys; "
                "import mutmut.__main__ as _mutmut_main; "
                "_mutmut_cwd = os.getcwd(); "
                f"os.chdir({project_root_literal}); "
                "_mutmut_main.ensure_config_loaded(); "
                "os.chdir(_mutmut_cwd); "
                "sys.argv = ['polylogue', *sys.argv[1:]]; "
                "runpy.run_module('polylogue', run_name='__main__')"
            ),
        ] + args

    # Open PTY
    master_fd, slave_fd = os.openpty()
    os.set_blocking(master_fd, False)

    start_time = time.time()
    output_chunks: list[bytes] = []

    try:
        # Start subprocess in PTY
        process = subprocess.Popen(
            command,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=clean_env,
            cwd=project_root,
            preexec_fn=os.setsid,  # Create new session so PTY works correctly
            start_new_session=False,
        )

        # Close slave fd in parent (not needed anymore)
        os.close(slave_fd)

        # Read output from master fd until process completes
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                process.kill()
                raise TimeoutError(
                    f"Command timed out after {timeout}s: {' '.join(command)}"
                )

            try:
                chunk = os.read(master_fd, 4096)
                if chunk:
                    output_chunks.append(chunk)
            except OSError:
                # EOF on master fd when slave closes
                pass

            # Check if process finished
            returncode = process.poll()
            if returncode is not None:
                break

            # Small sleep to avoid busy-waiting
            time.sleep(0.01)

        duration = time.time() - start_time

    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass

    # Combine output
    raw_output = b"".join(output_chunks)

    # Render using pyte with history to capture scrolled-off content
    screen = pyte.HistoryScreen(cols, rows, history=9999)
    screen.set_mode(pyte.modes.LNM)
    stream = pyte.Stream(screen)
    stream.feed(raw_output.decode("utf-8", errors="replace"))

    # Extract full output: scrollback history + visible screen
    def _chardict_to_str(chardict: Any) -> str:
        """Convert pyte history line (StaticDefaultDict of Char) to string."""
        if not chardict:
            return ""
        max_col = max(chardict.keys()) if chardict else 0
        return "".join(chardict[i].data for i in range(max_col + 1)).rstrip()

    history_lines = [_chardict_to_str(line) for line in screen.history.top]
    visible_lines = [line.rstrip() for line in screen.display]
    grid = history_lines + visible_lines

    return PtyResult(
        exit_code=process.returncode or 0,
        raw_output=raw_output,
        grid=grid,
        duration=duration,
    )


def sanitize_grid(
    grid: list[str],
    *,
    strip_timestamps: bool = True,
    strip_paths: bool = True,
    strip_durations: bool = True,
) -> list[str]:
    """Sanitize grid by replacing non-deterministic content.

    Args:
        grid: List of terminal lines
        strip_timestamps: Replace ISO timestamps and common date formats with <TIMESTAMP>
        strip_paths: Replace absolute paths with <PATH>
        strip_durations: Replace timing values (e.g., "1.234s") with <DURATION>

    Returns:
        Sanitized grid with non-deterministic content replaced
    """
    result = []

    for line in grid:
        # Strip timestamps (ISO 8601, common formats)
        if strip_timestamps:
            # ISO 8601: 2026-03-15T12:34:56.123456Z or similar
            line = re.sub(
                r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?",
                "<TIMESTAMP>",
                line,
            )
            # Simple date formats: 2026-03-15
            line = re.sub(r"\d{4}-\d{2}-\d{2}", "<TIMESTAMP>", line)
            # Time formats: 12:34:56
            line = re.sub(r"\d{2}:\d{2}:\d{2}", "<TIME>", line)

        # Strip absolute paths
        if strip_paths:
            # Unix-like paths
            line = re.sub(r"/[a-zA-Z0-9/_\-\.]+", "<PATH>", line)
            # Windows paths
            line = re.sub(r"[A-Za-z]:\\[a-zA-Z0-9\\_\-\.]+", "<PATH>", line)
            # Home directory shortcuts
            line = re.sub(r"~[a-zA-Z0-9/_\-\.]*", "<HOME>", line)

        # Strip durations (e.g., "1.234s", "1.234ms", "123s")
        if strip_durations:
            # Match: <digits>.<digits>ms/s or <digits>ms/s
            line = re.sub(r"\d+(?:\.\d+)?\s*m?s\b", "<DURATION>", line)

        result.append(line)

    return result


def grid_to_text(grid: list[str], *, strip_trailing: bool = True) -> str:
    """Convert grid to text representation.

    Args:
        grid: List of terminal lines
        strip_trailing: Remove trailing whitespace and empty lines

    Returns:
        Multi-line string representation
    """
    if strip_trailing:
        # Strip trailing whitespace from each line
        lines = [line.rstrip() for line in grid]
        # Remove trailing empty lines
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)
    else:
        return "\n".join(grid)
