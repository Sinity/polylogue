"""Generate VHS tape files (and optional GIF captures) for visual evidence.

This is the thin operator entrypoint over the tape engine in
``devtools.visual_vhs``. It writes one ``.tape`` file per default visual
evidence spec and, with ``--capture``, drives the ``vhs`` binary to render
the matching ``.gif`` files against the currently active archive.

The README documents ``devtools render visual-tapes --capture`` as the single
command that regenerates the demo screencast media, so the first-contact GIF
stays reproducible instead of being a committed binary that bitrots.
"""

from __future__ import annotations

import argparse
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
        help="verify the tape specs generate cleanly without writing any files",
    )
    args = parser.parse_args(argv)

    specs = default_tape_specs()

    if args.check:
        tapes = generate_all_tapes(specs)
        print(f"visual-tapes: {len(tapes)} tape specs generate cleanly")
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


if __name__ == "__main__":
    raise SystemExit(main())
