"""Re-exec a devtools module entry point into its invoking lane's own venv.

``devtools/click_dispatch.py:main()`` already does this for the primary
``devtools <args>`` command surface (``_run_from_invoking_lane``): a linked
worktree without its own interpreter on ``PATH`` would otherwise run under
whatever ``python`` it inherited (the coordinator's), and a coordinator's
editable install resolves ``polylogue``/``devtools`` from the WRONG checkout.

That routing lives behind ``devtools/__main__.py`` -> ``click_dispatch.main()``
only. Any entry point invoked as a *direct* module (``python -m
devtools.<name>``), bypassing that dispatcher, inherits the same hazard
without the same fix. ``.beads-hooks/pre-push`` is exactly this case: it runs
``python -m devtools.pre_push_gate "$UPDATES_FILE"`` directly, so a lane
worktree whose git hook subprocess inherits the coordinator's ``python``/
``PATH``/``VIRTUAL_ENV`` samples its checkout-provenance fingerprint
(``assert_polylogue_matches_checkout``) under the WRONG interpreter -- one
that never matches the receipt a correctly-routed ``devtools verify --quick``
wrote (via ``click_dispatch``), so the environment fingerprint in
``_has_compatible_quick_receipt`` always mismatches and every push reruns the
full quick gate instead of reusing the exact-match receipt.

This module is the shared, single-source re-exec used by both entry points,
mirroring the shape of ``devtools.checkout_guard`` (one shared resolver, not
independent hand-rolled comparisons at each call site).
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from devtools.checkout_guard import find_git_worktree_root


def lane_python_for(worktree: Path) -> Path | None:
    """Return the worktree's own venv interpreter, if one has been provisioned."""
    candidate = worktree / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def reexec_into_lane(module: str, argv: Sequence[str], *, cwd: Path | None = None) -> int | None:
    """Re-exec ``python -m <module> <argv>`` under the invoking lane's own venv.

    Returns ``None`` when the current interpreter already IS the lane's own
    venv interpreter (or no lane worktree applies, or none is provisioned
    yet), meaning the caller should continue running in-process. Returns an
    int exit code once a re-exec subprocess has run, so the caller returns
    that directly instead of continuing -- the identity check here (path
    equality against ``lane_python``, not a resolved/real path) is exactly
    what makes a second invocation, now running as that same lane python,
    see itself as already-matching and skip a further re-exec. This is the
    sole re-exec decision; every direct-module entry point that needs lane
    routing calls this instead of re-deriving its own comparison.
    """
    worktree = find_git_worktree_root(cwd or Path.cwd())
    if worktree is None:
        return None
    lane_python = lane_python_for(worktree)
    if lane_python is None or Path(sys.executable).absolute() == lane_python.absolute():
        return None
    from devtools.lane_init import lane_command_env

    return subprocess.run(
        [str(lane_python), "-m", module, *argv],
        cwd=worktree,
        env=lane_command_env(worktree),
        check=False,
    ).returncode
