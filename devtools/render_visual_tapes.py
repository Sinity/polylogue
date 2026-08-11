"""Generate VHS tape files (and optional GIF captures) for visual evidence.

This is the thin operator entrypoint over the tape engine in
``devtools.visual_vhs``. It writes one ``.tape`` file per default visual
evidence spec and, with ``--capture``, drives the ``vhs`` binary to render
the matching ``.gif`` files against the currently active archive.

The README documents ``devtools render visual-tapes --capture`` as the single
command that regenerates the demo screencast media, so the first-contact GIF
stays reproducible instead of being a committed binary that bitrots.

``--check`` confirms every default spec still generates cleanly and
byte-compares each generated tape against its committed counterpart under
``docs/examples/visual-tapes/``.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

from devtools.visual_vhs import (
    VHSTape,
    check_vhs_available,
    default_tape_specs,
    generate_all_tapes,
    run_vhs_capture,
)

DEFAULT_OUTPUT_DIR = ".local/visual-tapes"
COMMITTED_TAPES_DIR = Path("docs/examples/visual-tapes")


def committed_tape_drift(
    tapes: dict[str, str], *, committed_dir: Path = COMMITTED_TAPES_DIR
) -> dict[str, tuple[str | None, str]]:
    """Return generated tapes that differ from the committed visual surface."""
    drift: dict[str, tuple[str | None, str]] = {}
    for name, generated in tapes.items():
        committed_path = committed_dir / f"{name}.tape"
        committed_text = committed_path.read_text(encoding="utf-8") if committed_path.exists() else None
        if committed_text != generated:
            drift[name] = (committed_text, generated)
    return drift


def _print_drift(drift: dict[str, tuple[str | None, str]], *, committed_dir: Path = COMMITTED_TAPES_DIR) -> None:
    for name, (committed_text, generated) in sorted(drift.items()):
        if committed_text is None:
            print(f"visual-tapes: {name}: no committed tape at {committed_dir / f'{name}.tape'}", file=sys.stderr)
            continue
        print(
            f"visual-tapes: {name}: generated content differs from {committed_dir / f'{name}.tape'}",
            file=sys.stderr,
        )
        diff = difflib.unified_diff(
            committed_text.splitlines(keepends=True),
            generated.splitlines(keepends=True),
            fromfile=f"committed/{name}.tape",
            tofile=f"generated/{name}.tape",
        )
        sys.stderr.writelines(diff)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="devtools render visual-tapes",
        description="Write VHS tape files and optionally capture GIFs for the default visual evidence specs.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for generated .tape/.gif files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="run the 'vhs' binary to render .gif files from the generated tapes",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify generated tapes match the committed visual surface without writing files",
    )
    args = parser.parse_args(argv)

    specs = default_tape_specs()

    if args.check:
        tapes = generate_all_tapes(specs)
        drift = committed_tape_drift(tapes)
        if drift:
            print(f"visual-tapes: {len(drift)} committed tape(s) out of sync", file=sys.stderr)
            _print_drift(drift)
            return 1
        print(f"visual-tapes: {len(tapes)} committed tape(s) match their generated spec output")
        return 0

    output_dir = Path(args.output_dir)
    generate_all_tapes(specs, output_dir=output_dir)
    print(f"visual-tapes: wrote {len(specs)} .tape files to {output_dir}")

    if args.capture:
        if not check_vhs_available():
            # GIF capture is an optional, best-effort step: the tapes are the
            # committed artifact and were written above. A missing `vhs` binary
            # is a soft skip, not a failure, so the tapes still count as success.
            print(
                "visual-tapes: 'vhs' binary not found in PATH; tapes written, GIF capture skipped",
                file=sys.stderr,
            )
            return 0
        failures: list[str] = []
        for spec in specs:
            tape_path = output_dir / VHSTape(spec_name=spec.name, content="").filename
            gif_path = output_dir / f"{spec.name}.gif"
            if not run_vhs_capture(tape_path, gif_path):
                failures.append(spec.name)
        if failures:
            print(
                f"visual-tapes: capture failed for {', '.join(failures)}",
                file=sys.stderr,
            )
            return 1
        print(f"visual-tapes: captured {len(specs)} GIFs in {output_dir}")

    return 0


def generated_surface_main(argv: list[str] | None = None) -> int:
    """Render/check the committed tape files as a generated repository surface."""
    args = list(argv or [])
    if "--check" in args:
        return main(args)
    return main(["--output-dir", str(COMMITTED_TAPES_DIR), *args])


if __name__ == "__main__":
    raise SystemExit(main())
