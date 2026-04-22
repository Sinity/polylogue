"""Render the proof-obligation verification catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from devtools.command_catalog import control_plane_command
from devtools.render_support import write_if_changed
from polylogue.proof.catalog import VerificationCatalog, build_verification_catalog
from polylogue.proof.rendering import build_catalog_markdown


def _quality_failed(catalog: VerificationCatalog) -> bool:
    return any(check.status.value == "error" for check in catalog.quality_checks)


def _print_quality_errors(catalog: VerificationCatalog) -> None:
    for check in catalog.quality_checks:
        if check.status.value != "error":
            continue
        print(f"render-verification-catalog: quality error: {check.name}: {check.summary}", file=sys.stderr)
        for detail in check.details[:10]:
            print(f"  - {detail}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render docs/verification-catalog.md from proof registries.")
    parser.add_argument(
        "--output",
        default="docs/verification-catalog.md",
        help="Output file path or '-' for stdout (default: docs/verification-catalog.md)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the catalog is out of sync or self-quality checks fail.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the full catalog payload as JSON.")
    args = parser.parse_args(argv)

    catalog = build_verification_catalog()
    if args.json:
        json.dump(catalog.to_payload(), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 1 if args.check and _quality_failed(catalog) else 0

    rendered = build_catalog_markdown(catalog)
    if not rendered.endswith("\n"):
        rendered += "\n"

    if args.output == "-":
        if args.check:
            print("render-verification-catalog: --check does not support --output -", file=sys.stderr)
            return 2
        sys.stdout.write(rendered)
        return 0

    output_path = Path(args.output).expanduser()
    if args.check:
        try:
            current = output_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            current = ""
        failed = False
        if current != rendered:
            print(f"render-verification-catalog: out of sync: {output_path}", file=sys.stderr)
            print(
                "render-verification-catalog: run: "
                f"{control_plane_command('render-verification-catalog', '--output', str(output_path))}",
                file=sys.stderr,
            )
            failed = True
        if _quality_failed(catalog):
            _print_quality_errors(catalog)
            failed = True
        if failed:
            return 1
        print(f"render-verification-catalog: sync OK: {output_path}")
        return 0

    write_if_changed(output_path, rendered)
    if _quality_failed(catalog):
        _print_quality_errors(catalog)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
