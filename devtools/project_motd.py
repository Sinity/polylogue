"""Render a development-shell MOTD for Polylogue."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from devtools.generated_surfaces import GENERATED_SURFACES


@dataclass(frozen=True, slots=True)
class CheckStatus:
    label: str
    command: tuple[str, ...]


CHECKS: tuple[CheckStatus, ...] = tuple(
    CheckStatus(surface.label, (*surface.command, "--check")) for surface in GENERATED_SURFACES
)


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


def last_commit_subject(cwd: Path) -> str:
    lines = git_lines(cwd, "log", "-1", "--pretty=%s")
    return lines[0] if lines else "no commits"


def run_check(cwd: Path, check: CheckStatus) -> str:
    proc = subprocess.run(check.command, cwd=cwd, check=False, capture_output=True, text=True)
    return "ok" if proc.returncode == 0 else "stale"


def status_snapshot(cwd: Path) -> dict[str, object]:
    version = read_version(cwd / "pyproject.toml")
    branch, staged, changed, untracked = git_status_summary(cwd)
    generated_surfaces = {check.label: run_check(cwd, check) for check in CHECKS}
    stale_surfaces = [label for label, status in generated_surfaces.items() if status != "ok"]
    return {
        "project": "polylogue",
        "version": version,
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
            "discover": "python -m devtools --list-commands --json",
            "status": "python -m devtools status --json",
            "render_all_check": "python -m devtools render-all --check",
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
    return f"dirty · {staged} staged · {modified} modified · {untracked} untracked"


def summarize_generated_surfaces(generated_surfaces: dict[str, object]) -> str:
    total = len(generated_surfaces)
    stale = [label for label, status in generated_surfaces.items() if status != "ok"]
    if not stale:
        return f"{total}/{total} clean"
    fresh = total - len(stale)
    return f"{fresh}/{total} clean · stale: {', '.join(stale)}"


def render_motd(cwd: Path) -> str:
    snapshot = status_snapshot(cwd)
    changes = snapshot["changes"]
    assert isinstance(changes, dict)
    generated_surfaces = snapshot["generated_surfaces"]
    assert isinstance(generated_surfaces, dict)

    commands = snapshot["commands"]
    assert isinstance(commands, dict)
    header = f"Polylogue · {snapshot['branch']} · v{snapshot['version']}"
    rows = [
        ("worktree", summarize_worktree(changes)),
        ("generated", summarize_generated_surfaces(generated_surfaces)),
        ("recent", str(snapshot["last_commit"])),
        ("", ""),
        ("next", str(commands["render_all_check"])),
        ("", str(commands["test_baseline"])),
        ("", "ruff check polylogue tests devtools"),
        ("", str(commands["discover"])),
        ("", ""),
        ("local", ".cache/ · .local/ · result*"),
    ]

    label_width = max(len(label) for label, _ in rows if label)
    indent = " " * (label_width + 2)
    body_width = max(
        len(f"{label.ljust(label_width)}  {value}" if label else f"{indent}{value}" if value else "")
        for label, value in rows
    )
    frame_width = max(body_width + 2, len(header) + 2)
    top_rule = f"╭─ {header} " + "─" * max(frame_width - len(header) - 3, 0) + "╮"
    bottom_rule = f"╰{'─' * (len(top_rule) - 2)}╯"
    lines = [top_rule]
    for label, value in rows:
        if label:
            body = f"{label.ljust(label_width)}  {value}"
        elif value:
            body = f"{indent}{value}"
        else:
            body = ""
        lines.append(f"│ {body.ljust(frame_width)} │")
    lines.append(bottom_rule)
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
