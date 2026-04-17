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
from typing import TextIO

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


def status_snapshot(cwd: Path, *, verify_generated: bool = False) -> dict[str, object]:
    version = read_version(cwd / "pyproject.toml")
    branch, staged, changed, untracked = git_status_summary(cwd)
    revision = git_short_revision(cwd)
    if verify_generated:
        generated_surfaces = {surface.label: run_check(cwd, surface) for surface in GENERATED_SURFACES}
    else:
        generated_surfaces = {surface.label: "unchecked" for surface in GENERATED_SURFACES}
    stale_surfaces = [label for label, status in generated_surfaces.items() if status == "stale"]
    unchecked_surfaces = [label for label, status in generated_surfaces.items() if status == "unchecked"]
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
        "unchecked_surfaces": unchecked_surfaces,
        "generated_checked": verify_generated,
        "last_commit": last_commit_subject(cwd),
        "commands": {
            "discover": control_plane_command("--list-commands", "--json"),
            "status": control_plane_command("status", "--json"),
            "render_all_check": control_plane_command("render-all", "--check"),
            "verify_quick": control_plane_command("verify", "--quick"),
            "build_package": control_plane_command("build-package"),
            "test_baseline": "pytest -q --ignore=tests/integration",
        },
        "local_state": {
            "cache": ".cache/",
            "outputs": ".local/",
            "root_residents": [".venv/", ".direnv/"],
            "preferred_build_out_link": ".local/result",
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
    unchecked = [label for label, status in generated_surfaces.items() if status == "unchecked"]
    if len(unchecked) == total:
        return f"{total}/{total} generated unchecked"
    stale = [label for label, status in generated_surfaces.items() if status != "ok"]
    if not stale:
        return f"{total}/{total} generated clean"
    fresh = total - len(stale)
    return f"{fresh}/{total} generated clean · stale: {', '.join(stale)}"


def summarize_local_state(local_state: dict[str, object]) -> str:
    root_residents = local_state["root_residents"]
    assert isinstance(root_residents, list)
    keep_roots = " ".join(str(item) for item in root_residents)
    return (
        f"keep {keep_roots} · cache {local_state['cache']} · "
        f"outputs {local_state['outputs']} · build {local_state['preferred_build_out_link']}"
    )


def use_color(stream: TextIO = sys.stdout) -> bool:
    return stream.isatty() and os.environ.get("NO_COLOR") is None and os.environ.get("TERM") not in {None, "dumb"}


def style(text: str, *codes: str, stream: TextIO = sys.stdout) -> str:
    if not use_color(stream) or not codes:
        return text
    return "".join(codes) + text + ANSI_RESET


def style_worktree(summary: str, *, stream: TextIO) -> str:
    if summary == "clean":
        return style(summary, ANSI_GREEN, stream=stream)
    return style(summary, ANSI_YELLOW, ANSI_BOLD, stream=stream)


def style_generated(summary: str, *, stream: TextIO) -> str:
    if "stale:" in summary:
        return style(summary, ANSI_RED, ANSI_BOLD, stream=stream)
    if "unchecked" in summary:
        return style(summary, ANSI_YELLOW, stream=stream)
    return style(summary, ANSI_GREEN, stream=stream)


def render_motd(cwd: Path, *, verify_generated: bool = False, stream: TextIO = sys.stdout) -> str:
    snapshot = status_snapshot(cwd, verify_generated=verify_generated)
    changes = snapshot["changes"]
    assert isinstance(changes, dict)
    generated_surfaces = snapshot["generated_surfaces"]
    assert isinstance(generated_surfaces, dict)

    commands = snapshot["commands"]
    assert isinstance(commands, dict)
    local_state = snapshot["local_state"]
    assert isinstance(local_state, dict)
    revision = str(snapshot["revision"])
    dirty = summarize_worktree(changes) != "clean"
    display_version = f"v{snapshot['version']}+{revision}"
    if dirty:
        display_version += "-dirty"

    label_width = len("generated")
    rows = [
        ("worktree", style_worktree(summarize_worktree(changes), stream=stream)),
        ("generated", style_generated(summarize_generated_surfaces(generated_surfaces), stream=stream)),
        ("head", style(str(snapshot["last_commit"]), ANSI_DIM, stream=stream)),
        (
            "ready",
            style(
                f"{commands['render_all_check']} · {commands['verify_quick']} · {commands['build_package']}",
                ANSI_GREEN,
                stream=stream,
            ),
        ),
        ("test", style(str(commands["test_baseline"]), ANSI_GREEN, stream=stream)),
        ("roots", style(summarize_local_state(local_state), ANSI_DIM, stream=stream)),
    ]

    header = "  ".join(
        [
            style("Polylogue", ANSI_BOLD, ANSI_CYAN, stream=stream),
            style(str(snapshot["branch"]), ANSI_BOLD, stream=stream),
            style(display_version, ANSI_DIM if not dirty else ANSI_YELLOW, stream=stream),
        ]
    )
    lines = [header]
    for label, value in rows:
        lines.append(f"{style(label.ljust(label_width), ANSI_CYAN, stream=stream)}  {value}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the Polylogue devshell MOTD.")
    parser.add_argument("--cwd", default=".", help="Repository root (default: current directory)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable status instead of the MOTD.")
    parser.add_argument("--stderr", action="store_true", help="Write the MOTD to stderr instead of stdout.")
    parser.add_argument(
        "--verify-generated",
        action="store_true",
        help="Run generated-surface drift checks before rendering status.",
    )
    args = parser.parse_args(argv)

    cwd = Path(args.cwd).resolve()
    try:
        if args.json:
            json.dump(status_snapshot(cwd, verify_generated=args.verify_generated), sys.stdout, indent=2)
            sys.stdout.write("\n")
            return 0
        output_stream = sys.stderr if args.stderr else sys.stdout
        rendered = render_motd(cwd, verify_generated=args.verify_generated, stream=output_stream)
    except FileNotFoundError as exc:
        print(f"devtools status: missing file: {exc.filename}", file=sys.stderr)
        return 1

    output_stream.write(rendered)
    if os.environ.get("TERM"):
        output_stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
