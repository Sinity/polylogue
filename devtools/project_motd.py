"""Render a development-shell MOTD for Polylogue."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from devtools.command_catalog import control_plane_command
from devtools.generated_surfaces import GENERATED_SURFACES

ANSI_RESET = "\x1b[0m"
ANSI_BOLD = "\x1b[1m"
ANSI_DIM = "\x1b[2m"
ANSI_CYAN = "\x1b[36m"
ANSI_GREEN = "\x1b[32m"
ANSI_YELLOW = "\x1b[33m"
ANSI_RED = "\x1b[31m"


def read_version(pyproject_path: Path) -> str:
    text = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    return match.group(1) if match else "unknown"


def git_lines(cwd: Path, *args: str) -> list[str]:
    proc = subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def git_status_summary(cwd: Path) -> tuple[str, int, int, int]:
    branch = "-"
    staged = 0
    changed = 0
    untracked = 0
    for line in git_lines(cwd, "status", "--short", "--branch"):
        if line.startswith("## "):
            branch = line[3:].split("...")[0]
            continue
        if line.startswith("??"):
            untracked += 1
            continue
        if line[0] != " ":
            staged += 1
        if len(line) > 1 and line[1] != " ":
            changed += 1
    return branch, staged, changed, untracked


def git_short_revision(cwd: Path) -> str:
    lines = git_lines(cwd, "rev-parse", "--short=8", "HEAD")
    return lines[0] if lines else "unknown"


def last_commit_subject(cwd: Path) -> str:
    lines = git_lines(cwd, "log", "-1", "--pretty=%s")
    return lines[0] if lines else "no commits"


def run_check(cwd: Path, surface) -> str:
    stdout = io.StringIO()
    stderr = io.StringIO()
    previous_cwd = Path.cwd()
    try:
        os.chdir(cwd)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = surface.main(["--check"])
    except SystemExit as exc:
        result = int(exc.code or 0)
    finally:
        os.chdir(previous_cwd)
    return "ok" if result == 0 else "stale"


def status_snapshot(cwd: Path) -> dict[str, object]:
    version = read_version(cwd / "pyproject.toml")
    branch, staged, changed, untracked = git_status_summary(cwd)
    revision = git_short_revision(cwd)
    generated_surfaces = {surface.label: run_check(cwd, surface) for surface in GENERATED_SURFACES}
    stale_surfaces = [label for label, status in generated_surfaces.items() if status != "ok"]
    return {
        "project": "polylogue",
        "version": version,
        "revision": revision,
        "branch": branch,
        "changes": {
            "staged": staged,
            "modified": changed,
            "untracked": untracked,
        },
        "generated_surfaces": generated_surfaces,
        "stale_surfaces": stale_surfaces,
        "last_commit": last_commit_subject(cwd),
        "commands": {
            "discover": control_plane_command("--list-commands", "--json"),
            "status": control_plane_command("status", "--json"),
            "render_all_check": control_plane_command("render-all", "--check"),
            "test_baseline": "pytest -q --ignore=tests/integration",
        },
        "local_state": {
            "cache": ".cache/",
            "outputs": ".local/",
            "tool_owned": [".venv/", ".direnv/", "result*"],
        },
    }


def summarize_worktree(changes: dict[str, object]) -> str:
    staged = int(changes["staged"])
    modified = int(changes["modified"])
    untracked = int(changes["untracked"])
    if staged == 0 and modified == 0 and untracked == 0:
        return "clean"
    parts: list[str] = []
    if staged:
        parts.append(f"{staged} staged")
    if modified:
        parts.append(f"{modified} modified")
    if untracked:
        parts.append(f"{untracked} untracked")
    return "dirty · " + " · ".join(parts)


def summarize_generated_surfaces(generated_surfaces: dict[str, object]) -> str:
    total = len(generated_surfaces)
    stale = [label for label, status in generated_surfaces.items() if status != "ok"]
    if not stale:
        return f"{total}/{total} generated clean"
    fresh = total - len(stale)
    return f"{fresh}/{total} generated clean · stale: {', '.join(stale)}"


def use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None and os.environ.get("TERM") not in {None, "dumb"}


def style(text: str, *codes: str) -> str:
    if not use_color() or not codes:
        return text
    return "".join(codes) + text + ANSI_RESET


def style_worktree(summary: str) -> str:
    if summary == "clean":
        return style(summary, ANSI_GREEN)
    return style(summary, ANSI_YELLOW, ANSI_BOLD)


def style_generated(summary: str) -> str:
    if "stale:" in summary:
        return style(summary, ANSI_RED, ANSI_BOLD)
    return style(summary, ANSI_GREEN)


def render_motd(cwd: Path) -> str:
    snapshot = status_snapshot(cwd)
    changes = snapshot["changes"]
    assert isinstance(changes, dict)
    generated_surfaces = snapshot["generated_surfaces"]
    assert isinstance(generated_surfaces, dict)

    commands = snapshot["commands"]
    assert isinstance(commands, dict)
    revision = str(snapshot["revision"])
    dirty = summarize_worktree(changes) != "clean"
    display_version = f"v{snapshot['version']}+{revision}"
    if dirty:
        display_version += "-dirty"

    label_width = len("recent")
    rows = [
        ("repo", style_worktree(summarize_worktree(changes))),
        ("docs", style_generated(summarize_generated_surfaces(generated_surfaces))),
        ("recent", str(snapshot["last_commit"])),
        ("next", str(commands["render_all_check"])),
        ("", str(commands["test_baseline"])),
    ]
    indent = " " * (label_width + 2)

    header = "  ".join(
        [
            style("Polylogue", ANSI_BOLD, ANSI_CYAN),
            style(str(snapshot["branch"]), ANSI_BOLD),
            style(display_version, ANSI_DIM if not dirty else ANSI_YELLOW),
        ]
    )
    lines = [header]
    for label, value in rows:
        if label:
            lines.append(f"{style(label.ljust(label_width), ANSI_CYAN)}  {value}")
        else:
            lines.append(f"{indent}{value}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the Polylogue devshell MOTD.")
    parser.add_argument("--cwd", default=".", help="Repository root (default: current directory)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable status instead of the MOTD.")
    args = parser.parse_args(argv)

    cwd = Path(args.cwd).resolve()
    try:
        if args.json:
            json.dump(status_snapshot(cwd), sys.stdout, indent=2)
            sys.stdout.write("\n")
            return 0
        rendered = render_motd(cwd)
    except FileNotFoundError as exc:
        print(f"devtools status: missing file: {exc.filename}", file=sys.stderr)
        return 1

    sys.stdout.write(rendered)
    if os.environ.get("TERM"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
