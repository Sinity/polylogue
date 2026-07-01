"""Capture a current-curated CLI surface audit demo."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict


@dataclass(frozen=True, slots=True)
class AuditCommand:
    name: str
    argv: tuple[str, ...]
    notes: str


class ArchiveCounts(TypedDict):
    schema_version: int | None
    sessions: int | None
    messages: int | None


class CommandRecord(TypedDict):
    name: str
    argv: list[str]
    exit_code: int
    duration_ms: float
    stdout_bytes: int
    stderr_bytes: int
    stdout_lines: int
    stderr_lines: int
    notes: str


class AuditPayload(TypedDict):
    generated_at: str
    archive_root: str
    archive: ArchiveCounts
    include_unbounded_dialogue: bool
    commands: list[CommandRecord]


DEFAULT_COMMANDS: tuple[AuditCommand, ...] = (
    AuditCommand("root_help", ("polylogue", "--help"), "Dense but coherent root help."),
    AuditCommand("find_help", ("polylogue", "find", "--help"), "Short query workflow help."),
    AuditCommand("read_help", ("polylogue", "read", "--help"), "Projection, delivery, and read-view options."),
    AuditCommand(
        "read_views_json",
        ("polylogue", "read", "--views", "--format", "json"),
        "Machine-discoverable read-view registry.",
    ),
    AuditCommand("status_plain", ("polylogue", "--plain", "status"), "Compact archive/daemon status."),
    AuditCommand(
        "ops_status_json",
        ("polylogue", "--plain", "ops", "status", "--json"),
        "Machine status payload.",
    ),
    AuditCommand(
        "find_explain_read_json",
        (
            "polylogue",
            "--plain",
            "--format",
            "json",
            "--explain",
            "find",
            "repo:polylogue",
            "then",
            "read",
            "--view",
            "messages",
            "--limit",
            "1",
        ),
        "Query explain JSON for a messages read.",
    ),
    AuditCommand(
        "find_select_json",
        (
            "polylogue",
            "--plain",
            "--format",
            "json",
            "find",
            "repo:polylogue",
            "then",
            "select",
            "--limit",
            "3",
        ),
        "Bounded candidate identity selection.",
    ),
    AuditCommand(
        "read_dialogue_bounded_json",
        (
            "polylogue",
            "--plain",
            "find",
            "repo:polylogue",
            "then",
            "read",
            "--view",
            "dialogue",
            "--format",
            "json",
            "--limit",
            "1",
            "--max-tokens",
            "120",
        ),
        "Projection-bounded dialogue payload with omission accounting.",
    ),
    AuditCommand(
        "read_temporal_spec_json",
        (
            "polylogue",
            "--plain",
            "find",
            "repo:polylogue",
            "then",
            "read",
            "--view",
            "temporal,chronicle",
            "--spec",
            "--limit",
            "1",
        ),
        "Projection/render spec path.",
    ),
    AuditCommand(
        "facets_json",
        ("polylogue", "--plain", "facets", "--query", "repo:polylogue", "--format", "json"),
        "Archive-backed facets payload.",
    ),
)

UNBOUNDED_DIALOGUE_COMMAND = AuditCommand(
    "read_dialogue_unbounded_json",
    (
        "polylogue",
        "--plain",
        "find",
        "repo:polylogue",
        "then",
        "read",
        "--view",
        "dialogue",
        "--format",
        "json",
        "--limit",
        "1",
    ),
    "Opt-in diagnostic for the large unbounded dialogue payload.",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devtools workspace cli-surface-audit",
        description="Capture a current-curated CLI surface audit demo.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(".agent/demos/cli-surface-audit/current"))
    parser.add_argument("--archive-root", type=Path, default=Path.home() / ".local/share/polylogue")
    parser.add_argument("--include-unbounded-dialogue", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--json", action="store_true")
    return parser


def _archive_counts(archive_root: Path) -> ArchiveCounts:
    index_db = archive_root / "index.db"
    if not index_db.exists():
        return {"schema_version": None, "sessions": None, "messages": None}
    with sqlite3.connect(index_db) as conn:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        sessions = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        messages = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
    return {"schema_version": version, "sessions": sessions, "messages": messages}


def _commands(include_unbounded_dialogue: bool) -> tuple[AuditCommand, ...]:
    if include_unbounded_dialogue:
        return (*DEFAULT_COMMANDS, UNBOUNDED_DIALOGUE_COMMAND)
    return DEFAULT_COMMANDS


def _line_count(text: str) -> int:
    return text.count("\n") + (1 if text and not text.endswith("\n") else 0)


def _run_commands(commands: tuple[AuditCommand, ...], *, outdir: Path, timeout: int) -> list[CommandRecord]:
    summary: list[CommandRecord] = []
    for command in commands:
        started = time.perf_counter()
        proc = subprocess.run(command.argv, text=True, capture_output=True, timeout=timeout)
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        (outdir / f"{command.name}.stdout").write_text(proc.stdout, encoding="utf-8")
        (outdir / f"{command.name}.stderr").write_text(proc.stderr, encoding="utf-8")
        summary.append(
            {
                "name": command.name,
                "argv": list(command.argv),
                "exit_code": proc.returncode,
                "duration_ms": duration_ms,
                "stdout_bytes": len(proc.stdout.encode()),
                "stderr_bytes": len(proc.stderr.encode()),
                "stdout_lines": _line_count(proc.stdout),
                "stderr_lines": _line_count(proc.stderr),
                "notes": command.notes,
            }
        )
    return summary


def _readme(*, generated_at: str, archive_root: Path, archive: ArchiveCounts, commands: list[CommandRecord]) -> str:
    rows = [
        "| Command | Exit | Duration | Size | Notes |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for command in commands:
        rows.append(
            "| `{}` | {} | {:.0f} ms | {} bytes | {} |".format(
                " ".join(str(part) for part in command["argv"]),
                command["exit_code"],
                command["duration_ms"],
                command["stdout_bytes"],
                command["notes"],
            )
        )
    matrix = "\n".join(rows)
    return f"""# CLI Surface Audit

Generated: {generated_at}
Archive root: `{archive_root}`
Archive state: index schema v{archive["schema_version"]}, {archive["sessions"]} sessions, {archive["messages"]} messages

## What This Proves

This artifact is a live command audit over representative Polylogue CLI
surfaces: help, read-view discovery, daemon/archive status, query explain,
select, bounded dialogue read, temporal spec, and facets. It is not a complete
CLI certification; it is a dogfood slice that turns real command behavior into
concrete product work.

Raw command outputs are in `outputs/`. Timings, byte counts, exit codes, and
command notes are in `command-matrix.json`.

## Command Matrix

{matrix}

## Current Curation Policy

The default audit intentionally excludes the unbounded dialogue JSON export.
That diagnostic remains available with `--include-unbounded-dialogue`, but the
current demo shelf should prefer bounded dialogue output because it demonstrates
the product path an operator should actually use for large devloop sessions.

## Regeneration

```bash
devtools workspace cli-surface-audit --out-dir .agent/demos/cli-surface-audit/current
```
"""


def run_audit(
    *,
    out_dir: Path,
    archive_root: Path,
    include_unbounded_dialogue: bool,
    timeout: int,
) -> AuditPayload:
    out_dir = out_dir.expanduser()
    outputs = out_dir / "outputs"
    if outputs.exists():
        shutil.rmtree(outputs)
    outputs.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    archive = _archive_counts(archive_root)
    commands = _run_commands(_commands(include_unbounded_dialogue), outdir=outputs, timeout=timeout)
    payload: AuditPayload = {
        "generated_at": generated_at,
        "archive_root": str(archive_root),
        "archive": archive,
        "include_unbounded_dialogue": include_unbounded_dialogue,
        "commands": commands,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command-matrix.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "README.md").write_text(
        _readme(generated_at=generated_at, archive_root=archive_root, archive=archive, commands=commands),
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = run_audit(
            out_dir=args.out_dir,
            archive_root=args.archive_root,
            include_unbounded_dialogue=args.include_unbounded_dialogue,
            timeout=args.timeout,
        )
    except (OSError, sqlite3.Error, subprocess.SubprocessError) as exc:
        print(f"cli-surface-audit: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"captured {len(payload['commands'])} CLI command(s) under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
