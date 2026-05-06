"""Orchestrate the GitHub Pages documentation build.

Reads pages.toml, calls feeder render commands, assembles output
through Jinja2 templates, runs Pagefind, writes _site/.

Usage:
    devtools render-pages              Build _site/
    devtools render-pages --serve      Build and serve locally
    devtools render-pages --check      Verify _site/ matches sources
    devtools render-pages --watch      Rebuild on source changes
"""

from __future__ import annotations

import argparse
import http.server
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the polylogue GitHub Pages documentation site.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve _site/ on localhost:8080 after building.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify _site/ is in sync with sources (exit non-zero on drift).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Rebuild on source changes using watchfiles.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "_site",
        help="Output directory (default: _site/).",
    )
    parser.add_argument(
        "--skip-pagefind",
        action="store_true",
        help="Skip Pagefind indexing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output.",
    )
    return parser.parse_args(argv)


def _run_render_commands(verbose: bool = False) -> None:
    """Run feeder render-* commands to generate reference docs."""
    commands = [
        [sys.executable, "-m", "devtools", "render-cli-reference"],
        [sys.executable, "-m", "devtools", "render-devtools-reference"],
        [sys.executable, "-m", "devtools", "render-docs-surface"],
        [sys.executable, "-m", "devtools", "render-verification-catalog"],
        [sys.executable, "-m", "devtools", "render-quality-reference"],
        [sys.executable, "-m", "devtools", "render-topology-status"],
    ]
    for cmd in commands:
        if verbose:
            print(f"  Running: {' '.join(cmd)}", file=sys.stderr)
        try:
            subprocess.run(cmd, check=True, capture_output=not verbose, cwd=ROOT)
        except subprocess.CalledProcessError as e:
            print(f"  Warning: {' '.join(cmd)} failed: {e}", file=sys.stderr)


def _hash_directory(dir_path: Path) -> str:
    """Compute a rough hash of directory contents for drift detection."""
    import hashlib

    hasher = hashlib.sha256()
    for f in sorted(dir_path.rglob("*")):
        if f.is_file() and "pagefind" not in f.parts:
            hasher.update(f.read_bytes())
    return hasher.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    config_path = ROOT / "pages.toml"
    if not config_path.exists():
        print(f"Error: {config_path} not found", file=sys.stderr)
        return 1

    if args.check:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp) / "_site"
            from devtools.pages_builder import build_site

            build_site(config_path=config_path, output_dir=tmp_dir)
            current_hash = _hash_directory(args.output) if args.output.exists() else ""
            new_hash = _hash_directory(tmp_dir)
            if current_hash != new_hash:
                print("Error: _site/ is out of sync with sources. Run: devtools render-pages", file=sys.stderr)
                return 1
            print("OK: _site/ matches sources.")
            return 0

    print("Building GitHub Pages site...", file=sys.stderr)

    if args.verbose:
        print("  Running feeder render commands...", file=sys.stderr)
    _run_render_commands(verbose=args.verbose)

    if args.verbose:
        print("  Assembling pages...", file=sys.stderr)

    from devtools.pages_builder import build_site, build_site_with_pagefind

    if args.skip_pagefind:
        build_site(config_path=config_path, output_dir=args.output)
    else:
        build_site_with_pagefind(config_path=config_path, output_dir=args.output)

    print(f"  Site built: {args.output.resolve()}", file=sys.stderr)

    if args.serve:
        os.chdir(str(args.output))
        port = 8080
        handler = http.server.SimpleHTTPRequestHandler
        print(f"  Serving on http://localhost:{port}/", file=sys.stderr)
        http.server.HTTPServer(("127.0.0.1", port), handler).serve_forever()

    if args.watch:
        try:
            from watchfiles import run_process

            print("  Watching for changes...", file=sys.stderr)
            run_process(
                str(ROOT),
                target=sys.executable,
                args=("-m", "devtools", "render-pages", "--skip-pagefind"),
                callback=lambda changes: print(f"  Rebuilding ({len(changes)} changes)...", file=sys.stderr),
            )
        except ImportError:
            print("  Warning: watchfiles not installed. Install with: pip install watchfiles", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
