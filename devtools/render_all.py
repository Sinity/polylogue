"""Refresh or verify generated repository surfaces."""

from __future__ import annotations

import argparse
import sys

from devtools.generated_surfaces import GENERATED_SURFACES, GeneratedSurface


def _selected_surfaces(skip: set[str]) -> list[GeneratedSurface]:
    return [surface for surface in GENERATED_SURFACES if surface.name not in skip]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh or verify generated repository surfaces.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when any selected generated surface is out of sync.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=sorted(surface.name for surface in GENERATED_SURFACES),
        help="Skip a generated surface by name (repeatable).",
    )
    args = parser.parse_args(argv)

    selected = _selected_surfaces(set(args.skip))
    if not selected:
        print("render-all: no surfaces selected", file=sys.stderr)
        return 2

    exit_code = 0
    for surface in selected:
        result = surface.main(["--check"] if args.check else [])
        if result != 0:
            exit_code = result if exit_code == 0 else exit_code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
