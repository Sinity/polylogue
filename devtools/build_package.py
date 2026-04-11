"""Build the default Nix package into the repo-local output root."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DEFAULT_PACKAGE = ".#polylogue"
DEFAULT_OUT_LINK = ".local/result"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the default Nix package into .local/result.")
    parser.add_argument(
        "--package",
        default=DEFAULT_PACKAGE,
        help=f"Nix package reference to build (default: {DEFAULT_PACKAGE})",
    )
    parser.add_argument(
        "--out-link",
        default=DEFAULT_OUT_LINK,
        help=f"Output symlink path (default: {DEFAULT_OUT_LINK})",
    )
    args = parser.parse_args(argv)

    out_link = Path(args.out_link)
    out_link.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(["nix", "build", args.package, "--out-link", str(out_link)], check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
