"""Verify a freshly-run ``polylogue demo tour`` matches the committed
``docs/examples/demo-tour/`` evidence artifacts, modulo an explicit
volatile-field mask (polylogue-3tl.17).

Background
----------

``devtools render visual-tapes --check`` (polylogue-3tl.17) catches drift
between the tape *specs* and the committed ``.tape`` recipe files. It does
not catch drift between what ``polylogue demo tour`` actually *emits* at
runtime (transcript, report, per-step command output, recording tape) and
the committed copies of those emitted artifacts under
``docs/examples/demo-tour/`` -- the kit shipped a stale recorded tour twice
in 2026-07 (cwd-relative paths regressed once), caught only by a manual diff.

Volatile-field mask
--------------------

The tour is deterministic in every respect except wall-clock timing: seed
duration, verification duration, and per-step duration all render as
``<digits>.<digits>s`` in prose and Markdown tables. That is the ONLY masked
pattern -- claim text, exit codes, refs, anti-grep counts, command strings,
and the out-dir basename referenced inside the recording tape must match
exactly. A full-file skip would recreate the vacuity this gate exists to
kill.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

from devtools import repo_root as _get_root

ROOT = _get_root()
COMMITTED_DIR = ROOT / "docs" / "examples" / "demo-tour"

# Matches the committed fixture's own out-dir basename so freshly-run output
# references the same path the recording tape's commands were authored
# against (`cat polylogue-demo-tour/transcript.txt`, etc.).
_COMMITTED_OUT_DIR_BASENAME = "polylogue-demo-tour"

_DURATION_PATTERN = re.compile(r"\d+\.\d+s")


def mask_volatile(text: str) -> str:
    """Replace every wall-clock duration with a stable placeholder."""
    return _DURATION_PATTERN.sub("<duration>", text)


def _iter_comparable_relpaths(root: Path) -> list[str]:
    relpaths = ["transcript.txt", "report.md", "recording.tape"]
    command_output_dir = root / "command-output"
    if command_output_dir.is_dir():
        relpaths.extend(f"command-output/{p.name}" for p in sorted(command_output_dir.glob("*.txt")))
    return relpaths


def tour_freshness_diff(*, out_dir: Path) -> dict[str, tuple[str, str]]:
    """Run the demo tour into ``out_dir`` and return mismatches.

    ``out_dir``'s basename should be :data:`_COMMITTED_OUT_DIR_BASENAME` so
    the fresh recording tape's commands are directly comparable to the
    committed one.

    Returns ``{relpath: (committed_masked, fresh_masked)}`` for every file
    whose masked content differs, or whose fresh counterpart is missing
    entirely (fresh content reported as ``"<MISSING>"``).
    """
    from polylogue.demo.tour import run_demo_tour

    run_demo_tour(output_dir=out_dir)

    mismatches: dict[str, tuple[str, str]] = {}
    for relpath in _iter_comparable_relpaths(COMMITTED_DIR):
        committed_path = COMMITTED_DIR / relpath
        if not committed_path.exists():
            continue
        committed_masked = mask_volatile(committed_path.read_text(encoding="utf-8"))
        fresh_path = out_dir / relpath
        if not fresh_path.exists():
            mismatches[relpath] = (committed_masked, "<MISSING>")
            continue
        fresh_masked = mask_volatile(fresh_path.read_text(encoding="utf-8"))
        if committed_masked != fresh_masked:
            mismatches[relpath] = (committed_masked, fresh_masked)
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / _COMMITTED_OUT_DIR_BASENAME
        mismatches = tour_freshness_diff(out_dir=out_dir)

    if not mismatches:
        print("demo-tour-freshness: fresh tour output matches committed docs/examples/demo-tour/ (masked)")
        return 0

    print(f"demo-tour-freshness: {len(mismatches)} file(s) differ from the committed tour evidence:", file=sys.stderr)
    import difflib

    for relpath, (committed_masked, fresh_masked) in sorted(mismatches.items()):
        print(f"  {relpath}", file=sys.stderr)
        diff = difflib.unified_diff(
            committed_masked.splitlines(keepends=True),
            fresh_masked.splitlines(keepends=True),
            fromfile=f"committed/{relpath}",
            tofile=f"fresh/{relpath}",
        )
        sys.stderr.writelines(diff)
    print(
        "demo-tour-freshness: run 'polylogue demo tour --out-dir "
        f"{_COMMITTED_OUT_DIR_BASENAME} --force' and copy the regenerated files into "
        f"{COMMITTED_DIR.relative_to(ROOT)}/ to fix",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
