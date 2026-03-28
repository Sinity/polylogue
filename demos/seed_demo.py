#!/usr/bin/env python3
"""Seed a demo environment for screencasts and testing.

Thin wrapper around `polylogue demo --seed` for backwards compatibility.
Prefer using the CLI directly: polylogue demo --seed --env-only

Usage:
    python demos/seed_demo.py
    python demos/seed_demo.py -v
    eval $(python demos/seed_demo.py --env-only)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed a demo polylogue environment (wrapper for `polylogue demo --seed`)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--env-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cmd = ["polylogue", "demo", "--seed", "--count", str(args.count)]
    if args.output_dir:
        cmd.extend(["--output-dir", str(args.output_dir)])
    if args.env_only:
        cmd.append("--env-only")

    result = subprocess.run(cmd, capture_output=not args.verbose)
    if args.env_only and result.stdout:
        sys.stdout.buffer.write(result.stdout)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
