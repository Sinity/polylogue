#!/usr/bin/env python3
"""Seed a demo environment for screencasts and testing.

Sets up an isolated polylogue database by ingesting fixture conversations
through the real pipeline. Outputs the environment variables needed to
point polylogue at the demo database.

Usage:
    # Using real fixtures from tests/fixtures/real/
    python scripts/seed_demo.py

    # Using synthetic fallback (no real exports needed)
    python scripts/seed_demo.py --synthetic

    # Custom paths
    python scripts/seed_demo.py --fixtures-dir ./my-fixtures --output-dir /tmp/my-demo

    # Eval-friendly output for shell integration
    eval $(python scripts/seed_demo.py --synthetic --env-only)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _find_project_root() -> Path:
    """Walk up from this script to find the project root (where pyproject.toml lives)."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    raise RuntimeError("Cannot find project root (no pyproject.toml found)")


def seed_demo(
    *,
    fixtures_dir: Path,
    output_dir: Path,
    synthetic: bool = False,
    verbose: bool = False,
) -> dict[str, str]:
    """Seed a demo database and return the env vars to use it.

    Returns dict of environment variable name -> value.
    """
    # Set up isolated XDG paths BEFORE importing polylogue
    data_home = output_dir / "data"
    state_home = output_dir / "state"
    archive_root = output_dir / "archive"
    render_root = archive_root / "render"
    inbox_dir = output_dir / "inbox"

    for d in [data_home, state_home, archive_root, render_root, inbox_dir]:
        d.mkdir(parents=True, exist_ok=True)

    os.environ["XDG_DATA_HOME"] = str(data_home)
    os.environ["XDG_STATE_HOME"] = str(state_home)
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    os.environ["POLYLOGUE_RENDER_ROOT"] = str(render_root)
    os.environ["POLYLOGUE_FORCE_PLAIN"] = "1"

    # Now safe to import polylogue modules (they read env vars lazily)
    from polylogue.config import Config, Source
    from polylogue.pipeline.runner import run_sources

    # Determine fixture source
    if synthetic:
        # Import from scripts/fixtures/ relative to project root
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "synthetic", _find_project_root() / "scripts" / "fixtures" / "synthetic.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        fixture_output = output_dir / "synthetic-fixtures"
        mod.generate_synthetic_fixtures(fixture_output)
        source_dir = fixture_output
        if verbose:
            print(f"Generated synthetic fixtures in {fixture_output}", file=sys.stderr)
    else:
        source_dir = fixtures_dir
        if not source_dir.exists():
            print(f"Error: Fixtures directory not found: {source_dir}", file=sys.stderr)
            print("Use --synthetic to generate fixtures, or provide --fixtures-dir", file=sys.stderr)
            sys.exit(1)

    # Check for fixture content
    fixture_files = list(source_dir.rglob("*.json")) + list(source_dir.rglob("*.jsonl"))
    if not fixture_files:
        print(f"Warning: No fixture files found in {source_dir}", file=sys.stderr)
        if not synthetic:
            print("Hint: Symlink your real exports into tests/fixtures/real/", file=sys.stderr)
            print("  ln -s ~/Downloads/conversations.json tests/fixtures/real/chatgpt/", file=sys.stderr)
            print("Or use --synthetic for generated demo data", file=sys.stderr)
            sys.exit(1)

    # Discover sources from fixture subdirectories
    sources: list[Source] = []
    for subdir in sorted(source_dir.iterdir()):
        if subdir.is_dir() and any(subdir.rglob("*.json")) or any(subdir.rglob("*.jsonl")):
            sources.append(Source(name=subdir.name, path=subdir))

    if not sources:
        # Flat directory — treat the whole thing as inbox
        sources.append(Source(name="inbox", path=source_dir))

    if verbose:
        print(f"Discovered {len(sources)} sources:", file=sys.stderr)
        for s in sources:
            file_count = len(list(s.path.rglob("*.json"))) + len(list(s.path.rglob("*.jsonl")))
            print(f"  {s.name}: {file_count} files ({s.path})", file=sys.stderr)

    # Build config and run pipeline
    config = Config(
        archive_root=archive_root,
        render_root=render_root,
        sources=sources,
    )

    if verbose:
        print("Running pipeline...", file=sys.stderr)

    result = run_sources(
        config=config,
        stage="all",
        plan=None,
        ui=None,
        source_names=None,
    )

    if verbose:
        c = result.counts
        print(f"Ingested: {c.get('conversations', 0)} conversations, "
              f"{c.get('messages', 0)} messages", file=sys.stderr)
        print(f"Rendered: {c.get('rendered', 0)} files", file=sys.stderr)
        print(f"Duration: {result.duration_ms}ms", file=sys.stderr)
        if result.index_error:
            print(f"Index error: {result.index_error}", file=sys.stderr)

    # Return env vars needed to use this demo environment
    return {
        "XDG_DATA_HOME": str(data_home),
        "XDG_STATE_HOME": str(state_home),
        "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
        "POLYLOGUE_RENDER_ROOT": str(render_root),
        "POLYLOGUE_FORCE_PLAIN": "1",
    }


def main() -> None:
    project_root = _find_project_root()
    default_fixtures = project_root / "tests" / "fixtures" / "real"

    parser = argparse.ArgumentParser(
        description="Seed a demo polylogue environment for screencasts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Real fixtures (needs symlinked exports in tests/fixtures/real/)
  python scripts/seed_demo.py -v

  # Synthetic data (no real exports needed)
  python scripts/seed_demo.py --synthetic -v

  # Use in shell
  eval $(python scripts/seed_demo.py --synthetic --env-only)
  polylogue  # Shows demo stats
""",
    )
    parser.add_argument(
        "--fixtures-dir", type=Path, default=default_fixtures,
        help=f"Fixtures directory (default: {default_fixtures})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: /tmp/polylogue-demo/)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic fixtures instead of real exports",
    )
    parser.add_argument(
        "--env-only", action="store_true",
        help="Print only export statements (for eval)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="polylogue-demo-"))

    env_vars = seed_demo(
        fixtures_dir=args.fixtures_dir,
        output_dir=output_dir,
        synthetic=args.synthetic,
        verbose=args.verbose,
    )

    if args.env_only:
        for key, value in env_vars.items():
            print(f'export {key}="{value}"')
    else:
        print(f"\nDemo environment ready at: {output_dir}")
        print("\nTo use this environment, run:")
        print()
        for key, value in env_vars.items():
            print(f'  export {key}="{value}"')
        print()
        print("Then try:")
        print("  polylogue                    # Show archive stats")
        print("  polylogue run --preview      # Preview pipeline plan")
        print('  polylogue "error handling"   # Search conversations')
        print("  polylogue dashboard          # Launch TUI")


if __name__ == "__main__":
    main()
