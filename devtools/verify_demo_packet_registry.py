"""Verify every registered demo has a conforming Demo Finding Packet (polylogue-212.7).

Background
----------

The 212 demo portfolio manifest (``.agent/demos/registry.json``) lists every
demo prompt and its expected packet directory. This lint enumerates the
manifest and validates each entry's packet against the Demo Finding Packet
contract (``devtools.demo_packet``) -- catching a demo whose packet is
missing entirely, distinct from one whose packet exists but is malformed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from devtools.demo_packet import lint_demo_registry

DEFAULT_REGISTRY_PATH = Path(".agent/demos/registry.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help=f"path to the demo registry manifest (default: {DEFAULT_REGISTRY_PATH})",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    if not args.registry.exists():
        message = f"demo packet registry: manifest not found at {args.registry}"
        if args.json:
            print(json.dumps({"ok": False, "error": message}, indent=2))
        else:
            print(message)
        return 1

    result = lint_demo_registry(args.registry, repo_root=Path.cwd())

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.ok else 1

    if result.ok:
        print(f"demo packet registry: all {len(result.entry_results)} entries conform")
        return 0

    print("demo packet registry: violations found")
    for slug, validation in result.entry_results:
        if validation is None:
            print(f"  {slug}: packet directory missing")
            continue
        if not validation.ok:
            print(f"  {slug}: {validation.packet_dir}")
            for name in validation.missing_files:
                print(f"    missing file: {name}")
            for name in validation.missing_stanza_fields:
                print(f"    missing provenance stanza field: {name}")
            for name in validation.malformed_sections:
                print(f"    missing report.md section: {name}")
            for error in validation.errors:
                print(f"    {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
