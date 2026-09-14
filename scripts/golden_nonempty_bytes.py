#!/usr/bin/env python3
"""Non-empty companion to ``scripts/golden_find_bytes.py``.

``golden_find_bytes.py`` replays every documented ``polylogue find`` invocation
in ``docs/cli-reference.md``. That is the right acceptance set for "the
documented surface did not change", but on the seeded corpus most of the
documented examples that exercise the interesting code paths land on an *empty*
selection: the aggregate examples (``analyze count``, ``analyze by ...``) and
the ``id:``-scoped examples all match nothing, so a refactor can pass 37/37
without a single aggregate row, ranked hit, page boundary or attached-unit
projection ever being rendered. A renderer that drops every row, mislabels a
total, or loses a page's continuation is invisible to it.

This harness closes that gap with a hand-picked set of invocations chosen to
produce rows. They are hand-picked rather than extracted from documentation
precisely because the documentation does not contain them -- that absence is
the gap. Each case names one thing the documented set leaves unproven:

* list pages in all five output formats (``markdown``/``json``/``ndjson``/
  ``csv``/``yaml``) -- the row renderer's per-format branches;
* ``--limit`` and ``--limit``+``--offset`` -- the page boundary, ``total`` and
  ``next_offset``;
* ``--fields`` -- the row projection;
* ``analyze count``, ``analyze by origin``, ``analyze by day`` -- aggregate
  envelopes with rows in them, in text and JSON;
* a ranked search in text, JSON and CSV -- hit rows, ranks and snippets;
* ``--origin`` -- a filter that actually narrows;
* ``--sort``/``--reverse`` -- ordering;
* a ``messages where`` unit query -- the unit-row renderer;
* ``--sample`` -- the sampling window;
* ``with messages`` -- the attached-unit projection fold.

Usage, deliberately the same shape as the documented harness so a migration
step can run both::

    uv run python scripts/golden_nonempty_bytes.py OUTPUT.json
    uv run python scripts/golden_nonempty_bytes.py AFTER.json --compare BEFORE.json

``--compare`` reports the invocations whose exit code or bytes changed. It
normalizes three authority fields that vary per harness run on an unmodified
checkout -- ``run_id`` (a fresh UUID), ``generation_id`` (carries the archive
clone's inode) and ``archive_epoch`` (derived from it). Everything else is
compared byte-for-byte.

Determinism, the seeded workload and the frozen clock are all inherited from
``golden_find_bytes``; the two scripts are one harness pair and are committed
together, which is why this one imports that module's internals directly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from golden_find_bytes import (  # noqa: E402
    FROZEN_EPOCH,
    WORKLOAD,
    Runner,
    _frozen,
    _polylogue_modules,
    build_archive,
)

#: Hand-picked invocations that produce rows on the ``cli-mixed`` workload.
#: Each entry is the argv after ``polylogue``; ``--no-daemon`` is prepended by
#: the shared runner. Keep the list append-only where possible: removing a case
#: removes coverage, and the keys are the comparison identity.
CASES: tuple[tuple[str, ...], ...] = (
    # List pages: every output format the row renderer branches on.
    ("find",),
    ("find", "--format", "json"),
    ("find", "--format", "ndjson"),
    ("find", "--format", "csv"),
    ("find", "--format", "yaml"),
    # Page boundaries: total, limit clamping and next_offset.
    ("--limit", "3", "find", "--format", "json"),
    ("--limit", "2", "--offset", "1", "find", "--format", "json"),
    # Row projection.
    ("find", "--fields", "id,title,words", "--format", "json"),
    # Aggregates with rows in them -- the documented examples match nothing.
    ("analyze", "count"),
    ("analyze", "count", "--format", "json"),
    ("analyze", "by", "origin"),
    ("analyze", "by", "origin", "--format", "json"),
    ("analyze", "by", "day", "--format", "json"),
    # Ranked search: hit rows, ranks, snippets.
    ("find", "the", "--format", "json"),
    ("find", "the"),
    ("find", "the", "--format", "csv"),
    # A filter that narrows, and explicit ordering.
    ("--origin", "chatgpt", "find", "--format", "json"),
    ("--sort", "created", "find", "--format", "json"),
    ("--sort", "created", "--reverse", "find", "--format", "json"),
    # Unit rows, sampling, and the attached-unit projection fold.
    ("find", "messages where role:assistant", "--format", "json"),
    ("find", "--sample", "2", "--format", "json"),
    ("find", "with messages", "--format", "json"),
)

#: Authority fields that legitimately differ between two runs of this harness
#: on an unmodified checkout: a per-run UUID, and two values derived from the
#: archive clone's inode. Normalized for comparison only -- the recorded
#: goldens keep them verbatim.
_VOLATILE_FIELDS = ("run_id", "generation_id", "archive_epoch")
_VOLATILE_RE = re.compile(rf'"({"|".join(_VOLATILE_FIELDS)})": "[^"]*"')


def _normalize(text: str) -> str:
    return _VOLATILE_RE.sub(r'"\1": "<volatile>"', text)


def record(output: Path, workspace: Path | None = None) -> int:
    """Run every case and write exit code + exact stdout/stderr bytes."""

    root = (workspace or (REPO_ROOT / ".cache" / "golden-nonempty")).resolve()
    archive_root = root / "archive"
    home = root / "home"
    for path in (home / ".local/share", home / ".local/state", home / ".config", home / ".cache"):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    os.environ["POLYLOGUE_DAEMON"] = "off"

    clone = build_archive(archive_root)
    runner = Runner(archive_root=archive_root, home=home)

    import polylogue.cli.click_app  # noqa: F401  (populate sys.modules first)

    results: dict[str, object] = {}
    try:
        with _frozen(_polylogue_modules()):
            for argv in CASES:
                results[" ".join(argv)] = runner.run(argv)
    finally:
        close = getattr(clone, "close", None)
        if callable(close):
            close()

    payload = {
        "workload": WORKLOAD,
        "frozen_epoch": FROZEN_EPOCH,
        "source": "scripts/golden_nonempty_bytes.py:CASES",
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with_rows = sum(
        1
        for record in results.values()
        if isinstance(record, dict) and record.get("exit") == 0 and str(record.get("stdout", "")).strip()
    )
    print(f"wrote {len(results)} cases ({with_rows} exit 0 with output) to {output}")
    return 0


def compare(after: Path, before: Path) -> int:
    """Report the cases whose exit code or bytes changed. Nonzero when any did."""

    old = json.loads(before.read_text(encoding="utf-8"))["results"]
    new = json.loads(after.read_text(encoding="utf-8"))["results"]

    only_before = sorted(set(old) - set(new))
    only_after = sorted(set(new) - set(old))
    differing: list[str] = []
    for key in sorted(set(old) & set(new)):
        a, b = old[key], new[key]
        if a.get("exit") != b.get("exit") or any(
            _normalize(str(a.get(stream, ""))) != _normalize(str(b.get(stream, ""))) for stream in ("stdout", "stderr")
        ):
            differing.append(key)

    for key in only_before:
        print(f"REMOVED  {key}")
    for key in only_after:
        print(f"ADDED    {key}")
    for key in differing:
        a, b = old[key], new[key]
        print(f"DIFFERS  {key}")
        if a.get("exit") != b.get("exit"):
            print(f"    exit   {a.get('exit')} -> {b.get('exit')}")
        for stream in ("stdout", "stderr"):
            if _normalize(str(a.get(stream, ""))) != _normalize(str(b.get(stream, ""))):
                print(f"    {stream} before: {str(a.get(stream, ''))[:400]!r}")
                print(f"    {stream} after:  {str(b.get(stream, ''))[:400]!r}")

    changed = len(only_before) + len(only_after) + len(differing)
    print(f"{len(set(old) & set(new))} compared, {len(differing)} differing, {changed} changed overall")
    return 1 if changed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Non-empty golden bytes for the root query surface.")
    parser.add_argument("output", type=Path, help="path to write the golden JSON")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="directory for the seeded archive clone and fake HOME",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        metavar="BASELINE",
        help="after recording, diff against a previously recorded golden file",
    )
    args = parser.parse_args(argv)

    status = record(args.output, args.workspace)
    if status or args.compare is None:
        return status
    return compare(args.output, args.compare)


if __name__ == "__main__":
    raise SystemExit(main())
