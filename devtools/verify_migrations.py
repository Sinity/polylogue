"""Enforce the migration-completeness manifest.

Each migration declares paths, CLI commands, devtools commands, and
forbidden substrings that must vanish for it to count as complete.
This command is for active transitions only. Completed migrations should be
deleted from the manifest and replaced with semantic boundary checks when their
post-state remains important.

The default lint mode treats unstarted migrations as informational so landing
this manifest does not gate every PR. Use ``--strict <migration-name>`` for an
active transition that must block until its done detector passes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "plans" / "migrations.yaml"


def _strip_inline_comment(text: str) -> str:
    """Strip a trailing YAML comment (`` # ...``) from a value line.

    Only strips `` #`` when the ``#`` is outside any quoted string, so
    ``value: "foo # bar"`` is not truncated.
    """
    in_quotes = False
    quote_char: str | None = None
    for i, ch in enumerate(text):
        if ch in ('"', "'"):
            if not in_quotes:
                in_quotes = True
                quote_char = ch
            elif ch == quote_char:
                in_quotes = False
        elif ch == "#" and not in_quotes and i > 0 and text[i - 1] == " ":
            return text[:i].rstrip()
    return text


def parse_yaml(text: str) -> dict[str, Any]:
    """Tiny YAML reader for the migrations schema."""
    out: dict[str, Any] = {"migrations": {}, "completed": []}
    state = "top"
    cur_migration: str | None = None
    cur_field: str | None = None
    cur_subfield_dict: dict[str, str] | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        line = _strip_inline_comment(line)
        indent = len(line) - len(line.lstrip())
        stripped = line.lstrip()
        if indent == 0:
            if cur_subfield_dict is not None and cur_migration and cur_field:
                out["migrations"][cur_migration][cur_field].append(cur_subfield_dict)
                cur_subfield_dict = None
            if stripped == "migrations: {}":
                state = "migrations"
                cur_migration = None
                continue
            if stripped == "completed: []":
                state = "completed"
                cur_migration = None
                continue
            key = stripped.rstrip(":")
            if key in {"migrations", "completed"}:
                state = key
                cur_migration = None
            continue
        if state == "migrations":
            if indent == 2 and stripped.endswith(":"):
                cur_migration = stripped.rstrip(":")
                out["migrations"][cur_migration] = {}
                cur_field = None
            elif indent == 4 and ": " in stripped and cur_migration:
                k, _, v = stripped.partition(": ")
                out["migrations"][cur_migration][k] = _coerce(v.strip())
                cur_field = None
            elif indent == 4 and stripped.endswith(":") and cur_migration:
                cur_field = stripped.rstrip(":")
                out["migrations"][cur_migration][cur_field] = []
                cur_subfield_dict = None
            elif indent == 4 and stripped == "must_vanish_paths: []":
                # already handled by ": " path
                pass
            elif indent == 6 and stripped.startswith("- ") and cur_migration and cur_field:
                rest = stripped[2:]
                if ": " in rest:
                    if cur_subfield_dict is not None:
                        out["migrations"][cur_migration][cur_field].append(cur_subfield_dict)
                    cur_subfield_dict = {}
                    k, _, v = rest.partition(": ")
                    cur_subfield_dict[k] = _coerce(v.strip())
                else:
                    if cur_subfield_dict is not None:
                        out["migrations"][cur_migration][cur_field].append(cur_subfield_dict)
                        cur_subfield_dict = None
                    out["migrations"][cur_migration][cur_field].append(_coerce(rest.strip()))
            elif indent == 8 and ": " in stripped and cur_subfield_dict is not None:
                k, _, v = stripped.partition(": ")
                cur_subfield_dict[k] = _coerce(v.strip())
        elif state == "completed" and stripped.startswith("- "):
            out["completed"].append(_coerce(stripped[2:].strip()))
    if cur_subfield_dict is not None and cur_migration and cur_field:
        out["migrations"][cur_migration][cur_field].append(cur_subfield_dict)
    return out


def _coerce(value: str) -> Any:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value == "[]":
        return []
    return value


def _polylogue_subcommand_help() -> str:
    try:
        r = subprocess.run(["polylogue", "--help"], capture_output=True, text=True, timeout=15, check=False)
        return r.stdout + r.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _devtools_subcommand_list() -> str:
    try:
        r = subprocess.run(["devtools", "--list-commands"], capture_output=True, text=True, timeout=10, check=False)
        return r.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def check_migration(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    findings: dict[str, list[str]] = defaultdict(list)

    for rel in spec.get("must_vanish_paths", []) or []:
        if (ROOT / rel).exists():
            findings["surviving_paths"].append(rel)

    if spec.get("must_vanish_cli_commands"):
        help_text = _polylogue_subcommand_help()
        for cmd in spec.get("must_vanish_cli_commands", []) or []:
            # appearance in help suggests command still exists
            if f"  {cmd} " in help_text or f"  {cmd}\n" in help_text:
                findings["surviving_cli_commands"].append(cmd)

    if spec.get("must_vanish_devtools_commands"):
        listing = _devtools_subcommand_list()
        for cmd in spec.get("must_vanish_devtools_commands", []) or []:
            if cmd in listing:
                findings["surviving_devtools_commands"].append(cmd)

    for entry in spec.get("forbidden_substrings", []) or []:
        if not isinstance(entry, dict):
            continue
        glob = entry.get("location_glob", "")
        substring = entry.get("substring", "")
        if not glob or not substring:
            continue
        for path in ROOT.glob(glob):
            if "__pycache__" in path.parts:
                continue
            try:
                if substring in path.read_text():
                    findings["surviving_substrings"].append(f"{path.relative_to(ROOT)} contains {substring!r}")
            except OSError:
                continue

    final_findings = {k: v for k, v in findings.items() if v}
    pending = not final_findings
    has_constraints = bool(
        spec.get("must_vanish_paths")
        or spec.get("must_vanish_cli_commands")
        or spec.get("must_vanish_devtools_commands")
        or spec.get("forbidden_substrings")
    )

    return {
        "name": name,
        "issue": spec.get("issue", ""),
        "description": spec.get("description", ""),
        "complete": pending and has_constraints,
        "findings": final_findings,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yaml", type=Path, default=MANIFEST)
    p.add_argument("--strict", action="append", default=[], help="Migration names that must report zero findings.")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    manifest = parse_yaml(args.yaml.read_text())
    migration_names = set(manifest["migrations"].keys())
    unknown_strict = sorted(n for n in args.strict if n not in migration_names)
    if unknown_strict:
        msg = f"unknown --strict migration name(s): {unknown_strict}"
        print(f"[error] {msg}", file=sys.stderr)
        if args.json:
            json.dump({"blocking": True, "error": msg}, sys.stdout, indent=2)
            sys.stdout.write("\n")
        return 1
    results: list[dict[str, Any]] = []
    blocking = False
    for name, spec in manifest["migrations"].items():
        if not isinstance(spec, dict):
            continue
        result = check_migration(name, spec)
        results.append(result)
        if name in args.strict and any(result["findings"].values()):
            blocking = True

    if args.json:
        json.dump({"blocking": blocking, "migrations": results}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for result in results:
            tag = "[BLOCK]" if result["name"] in args.strict and any(result["findings"].values()) else "[info]"
            findings = result.get("findings", {})
            n = sum(len(v) for v in findings.values())
            status = "complete" if result["complete"] else f"{n} surviving"
            issue = str(result.get("issue") or "").strip()
            suffix = f" ({issue})" if issue else ""
            print(f"{tag} {result['name']}{suffix}: {status}")
            for kind, items in findings.items():
                for item in items[:5]:
                    print(f"    {kind}: {item}")
                if len(items) > 5:
                    print(f"    {kind}: ... and {len(items) - 5} more")
        print()
        print(f"blocking={blocking}")

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
