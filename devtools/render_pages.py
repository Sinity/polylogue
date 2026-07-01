"""Orchestrate the GitHub Pages documentation build.

Reads docs/site/pages.toml, calls feeder render commands, assembles output
through Jinja2 templates, runs Pagefind, writes .cache/site/.

Usage:
    devtools render pages              Build .cache/site/
    devtools render pages --serve      Build and serve locally
    devtools render pages --check      Verify sources render and links resolve
    devtools render pages --watch      Rebuild on source changes
"""

from __future__ import annotations

import argparse
import http.server
import os
import sys
from pathlib import Path

from devtools import repo_root as _get_root

ROOT = _get_root()


def _default_config_path() -> Path:
    return ROOT / "docs" / "site" / "pages.toml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the polylogue GitHub Pages documentation site.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve .cache/site/ on localhost:8080 after building.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the site sources render and generated local links resolve.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Rebuild on source changes using watchfiles.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".cache" / "site",
        help="Output directory (default: .cache/site/).",
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
    from devtools import (
        render_cli_reference,
        render_devtools_reference,
        render_docs_surface,
        render_quality_reference,
        render_topology_status,
    )

    renderers = [
        ("render cli-reference", render_cli_reference.main),
        ("render devtools-reference", render_devtools_reference.main),
        ("render docs-surface", render_docs_surface.main),
        ("render quality-reference", render_quality_reference.main),
        ("render topology-status", render_topology_status.main),
    ]
    for name, render in renderers:
        if verbose:
            print(f"  Running: devtools {name}", file=sys.stderr)
        try:
            result = render([])
        except Exception as exc:
            print(f"  Warning: devtools {name} failed: {exc}", file=sys.stderr)
            continue
        if result != 0:
            print(f"  Warning: devtools {name} failed with exit code {result}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    config_path = _default_config_path()
    if not config_path.exists():
        print(f"Error: {config_path} not found", file=sys.stderr)
        return 1

    if args.check:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp) / "_site"
            from devtools.pages_builder import build_site, validate_site_links

            build_site(config_path=config_path, output_dir=tmp_dir)
            broken_links = validate_site_links(tmp_dir)
            if broken_links:
                print("Error: generated site has broken local links:", file=sys.stderr)
                for link in broken_links[:20]:
                    print(f"  {link.page}: {link.href} -> {link.target}", file=sys.stderr)
                if len(broken_links) > 20:
                    print(f"  ... and {len(broken_links) - 20} more", file=sys.stderr)
                return 1
            print("OK: site sources render and generated local links resolve.")
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
                args=("-m", "devtools", "render pages", "--skip-pagefind"),
                callback=lambda changes: print(f"  Rebuilding ({len(changes)} changes)...", file=sys.stderr),
            )
        except ImportError:
            print("  Warning: watchfiles not installed. Install with: pip install watchfiles", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
