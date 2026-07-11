"""Validate one Demo Packet v2 directory and resolve every declared receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from devtools.demo_packet import DEFAULT_SCHEMA_PATH, PacketValidationResult, validate_packet


def _render_plain(result: PacketValidationResult) -> None:
    if result.ok:
        print(f"demo packet: conforming: {result.packet_dir}")
        return
    print(f"demo packet: violations found: {result.packet_dir}")
    for name in result.missing_files:
        print(f"  missing file: {name}")
    for name in result.missing_stanza_fields:
        print(f"  missing provenance stanza field: {name}")
    for name in result.malformed_sections:
        print(f"  missing report.md section: {name}")
    for error in result.schema_errors:
        print(f"  schema: {error}")
    for error in result.receipt_errors:
        print(f"  receipt: {error}")
    for error in result.errors:
        print(f"  {error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet_dir", type=Path, help="Demo Packet v2 directory")
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help=f"schema path (default: {DEFAULT_SCHEMA_PATH})",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    result = validate_packet(args.packet_dir, schema_path=args.schema)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _render_plain(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
