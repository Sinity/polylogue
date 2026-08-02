"""Staged-content secret-candidate scan for the pre-commit gate (polylogue-t9xd).

Leak-surfaces audit finding L5/L11 (2026-07-31): the existing
``.githooks/pre-commit`` ran ``ruff format``/``ruff check`` on staged
``*.py`` files only -- no gate anywhere scanned staged content, of any file
type, for credential-shaped spans before it became a git commit. This module
is what the hook now invokes: it scans the *staged (index) blob* of every
added/modified path (not the working tree, so a secret already reverted
locally but still staged is still caught), using the same
``scan_text_for_secret_candidates`` rules and never-log-the-literal
invariant as ``polylogue ops scan-secrets``.

This is explicitly the same detector wired into a second, un-related place
(see :mod:`polylogue.cli.read_views.base` for the render/export side of the
same fix) -- both are candidate-only triage aids, not a leak-prevention
boundary in their own right (``docs/security.md``, "Raw artifacts are not
content-redacted"). Warn-only by default: false positives on legitimate
content (API docs, sample configs, the scanner's own test fixtures) would be
disruptive to hard-block on unconditionally. Set
``POLYLOGUE_SECRET_SCAN_BLOCK=1`` to fail the commit on any finding instead
of only warning.

Exit codes distinguish "ran clean" (0), "found candidates, blocking per env
opt-in" (1), and "did not run" (2, e.g. import/subprocess failure) so the
calling hook can treat (2) as non-fatal instead of silently conflating it
with a real finding.
"""

from __future__ import annotations

import os
import subprocess
import sys

from polylogue.security.secret_scan import describe_secret_candidate_spans, scan_text_for_secret_candidates

# Paths that legitimately contain deliberate secret-shaped literals as test
# fixtures for the scanner itself -- scanning them would fire on every
# commit that touches the scanner's own test suite.
_SKIP_PATH_SUBSTRINGS = (
    "tests/unit/security/test_secret_scan",
    "tests/unit/security/test_precommit_scan",
)

#: Skip blobs above this size -- large/binary-ish staged content (lockfiles,
#: fixtures, images) is not worth decoding+scanning on every commit.
_MAX_SCAN_BYTES = 2_000_000


def _staged_blob_text(path: str) -> str | None:
    """Return the staged (index) content of ``path``, or ``None`` to skip it."""
    try:
        raw = subprocess.run(["git", "show", f":{path}"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    if len(raw) > _MAX_SCAN_BYTES:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan_staged_paths(paths: list[str]) -> int:
    """Scan the staged content of ``paths``; return the total candidate count.

    Prints a per-file, no-literal summary line to stderr for every path with
    a hit as a side effect.
    """
    total = 0
    for path in paths:
        if any(skip in path for skip in _SKIP_PATH_SUBSTRINGS):
            continue
        text = _staged_blob_text(path)
        if not text:
            continue
        spans = scan_text_for_secret_candidates(text)
        if not spans:
            continue
        total += len(spans)
        print(f"pre-commit: secret-scan: {path}: {describe_secret_candidate_spans(spans)}", file=sys.stderr)
    return total


def main(argv: list[str]) -> int:
    if not argv:
        return 0
    total = scan_staged_paths(argv)
    if total == 0:
        return 0
    print(
        f"pre-commit: secret-scan: {total} credential-shaped candidate span(s) found in staged "
        "content -- review before pushing. This is a candidate detector (pattern-shape + entropy), "
        "not proof of a real secret, and it never logs the matched text "
        "(polylogue/security/secret_scan.py).",
        file=sys.stderr,
    )
    if os.environ.get("POLYLOGUE_SECRET_SCAN_BLOCK") == "1":
        print("pre-commit: secret-scan: blocking commit (POLYLOGUE_SECRET_SCAN_BLOCK=1)", file=sys.stderr)
        return 1
    print(
        "pre-commit: secret-scan: not blocking (set POLYLOGUE_SECRET_SCAN_BLOCK=1 to make this fatal)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
