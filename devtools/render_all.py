"""Refresh or verify generated repository surfaces.

Each surface declares its input files via ``GeneratedSurface.inputs``.
Before rendering, a content hash of those files is compared against a
stored stamp (``.cache/.render-<name>-stamp``). If the hash matches, the
surface is skipped — its last render is still current.

Surfaces render in registry order because generators share process-global state.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import sys
from pathlib import Path

from devtools.generated_surfaces import GENERATED_SURFACES, GeneratedSurface

CACHE_DIR = Path(".cache")


def _surface_input_hash(surface: GeneratedSurface) -> str:
    """SHA-256 of all files matched by the surface's input globs."""
    h = hashlib.sha256()
    for pattern in surface.inputs:
        p = Path(pattern)
        if p.is_file():
            h.update(p.read_bytes())
        elif p.is_dir():
            for f in sorted(p.rglob("*.py")):
                with contextlib.suppress(OSError):
                    h.update(f.read_bytes())
            for f in sorted(p.rglob("*.yaml")):
                with contextlib.suppress(OSError):
                    h.update(f.read_bytes())
            for f in sorted(p.rglob("*.md")):
                with contextlib.suppress(OSError):
                    h.update(f.read_bytes())
        elif "*" in pattern or "?" in pattern:
            for f in sorted(Path().glob(pattern)):
                if f.is_file():
                    with contextlib.suppress(OSError):
                        h.update(f.read_bytes())
    return h.hexdigest()


def _stamp_path(name: str) -> Path:
    return CACHE_DIR / f".render-{name}-stamp"


def _is_fresh(surface: GeneratedSurface) -> bool:
    """Return True if the surface's rendered output is current."""
    inputs = getattr(surface, "inputs", ())
    if not inputs:
        return False
    stamp = _stamp_path(surface.name)
    if not stamp.exists():
        return False
    try:
        current_hash = _surface_input_hash(surface)
        return stamp.read_text().strip() == current_hash
    except OSError:
        return False


def _stamp(surface: GeneratedSurface) -> None:
    """Record the current input hash after a successful render."""
    if not getattr(surface, "inputs", ()):
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _stamp_path(surface.name).write_text(_surface_input_hash(surface) + "\n")


def _render_one(surface: GeneratedSurface, check: bool) -> int:
    """Render or check a single surface. Returns exit code."""
    if not check and _is_fresh(surface):
        print(f"render all: skip {surface.name} (inputs unchanged)", file=sys.stderr)
        return 0

    mode = "check" if check else "render"
    print(f"render all: {mode} {surface.name}", file=sys.stderr)
    result = surface.main(["--check"] if check else [])
    if result == 0 and not check:
        _stamp(surface)
    return result


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
        print("render all: no surfaces selected", file=sys.stderr)
        return 2

    exit_code = 0
    if args.check:
        for surface in selected:
            result = _render_one(surface, check=True)
            if result != 0:
                exit_code = result if exit_code == 0 else exit_code
        return exit_code

    # Render in registry order. Several generators temporarily change the process
    # working directory or stage shared files, so thread-level parallelism can
    # make unrelated surfaces resolve paths against the wrong directory.
    for surface in selected:
        result = _render_one(surface, check=False)
        if result != 0:
            exit_code = result if exit_code == 0 else exit_code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
