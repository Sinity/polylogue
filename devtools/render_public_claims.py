"""Render public-claims presets from FINDING assertions and 37t.14 verdict receipts."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from devtools.command_catalog import control_plane_command
from devtools.public_claims import (
    DEFAULT_COMPATIBILITY_PATH,
    DEFAULT_OUTPUT_DIR,
    build_repository_projection,
    rendered_artifacts,
)
from devtools.render_support import write_if_changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Markdown/JSON preset directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--compatibility-path",
        type=Path,
        default=DEFAULT_COMPATIBILITY_PATH,
        help=f"generated YAML compatibility view (default: {DEFAULT_COMPATIBILITY_PATH})",
    )
    parser.add_argument("--archive-root", type=Path, help="read public FINDING rows from this archive's user.db")
    parser.add_argument("--verdicts", type=Path, help="37t.14 JSON verdict receipt export")
    parser.add_argument("--check", action="store_true", help="exit non-zero when any generated artifact is out of sync")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable render report")
    args = parser.parse_args(argv)

    try:
        claims = build_repository_projection(archive_root=args.archive_root, verdicts_path=args.verdicts)
        artifacts = rendered_artifacts(
            claims,
            output_dir=args.output_dir,
            compatibility_path=args.compatibility_path,
        )
    except (OSError, ValueError, json.JSONDecodeError, sqlite3.Error) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"render public-claims: {exc}", file=sys.stderr)
        return 2

    changed = [
        path
        for path, rendered in artifacts.items()
        if (path.read_text(encoding="utf-8") if path.exists() else "") != rendered
    ]
    if args.check:
        ok = not changed
        if args.json:
            print(
                json.dumps(
                    {
                        "ok": ok,
                        "claim_count": len(claims),
                        "artifact_count": len(artifacts),
                        "out_of_sync": [str(path) for path in changed],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        elif ok:
            print("render public-claims: sync OK")
        else:
            print("render public-claims: out of sync:", file=sys.stderr)
            for path in changed:
                print(f"  - {path}", file=sys.stderr)
            print(
                f"render public-claims: run: {control_plane_command('render public-claims')}",
                file=sys.stderr,
            )
        return 0 if ok else 1

    for path, rendered in artifacts.items():
        write_if_changed(path, rendered)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": True,
                    "claim_count": len(claims),
                    "artifact_count": len(artifacts),
                    "written": [str(path) for path in changed],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
