"""End-to-end proof that the pre-push hook routes through the invoking lane's
own venv before sampling checkout provenance.

``.beads-hooks/pre-push`` invokes ``python -m devtools.pre_push_gate
"$UPDATES_FILE"`` directly -- bypassing ``devtools/click_dispatch.py:main()``,
which is the only place the ordinary ``devtools <cmd>`` surface re-execs into
a lane's own venv interpreter. A harness-created lane worktree can inherit a
coordinator's ``python``/``PATH``/``VIRTUAL_ENV``, so the hook's own checkout-
provenance sample (``assert_polylogue_matches_checkout``, driven by
``sys.executable``'s own path) would run under the WRONG interpreter and
never match the receipt a correctly-routed ``devtools verify --quick``
wrote -- every push then reruns the full quick gate instead of reusing an
exact-match receipt (the bug this test pins).

This spins up a real, independent linked worktree of the current checkout (so
production ``devtools``/``polylogue`` code runs for real, not a stub), gives
it its own venv symlink, and invokes the REAL module entry point
(``python -m devtools.pre_push_gate``) from a second, deliberately "foreign"
interpreter path whose own ``.venv`` parent differs from the lane root --
exactly the shape ``checkout_guard._python_environment_root`` flags as
belonging to another checkout. The fix must re-exec into the lane's own venv
exactly once before sampling provenance, after which a pre-seeded compatible
receipt reuses.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from devtools.verify_runs import worktree_fingerprint

ROOT = Path(__file__).resolve().parents[3]


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise AssertionError(f"git {args} failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}")
    return result


@pytest.fixture
def lane_worktree(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    """A real, independent linked worktree of THIS checkout at its current HEAD.

    Independent of whatever uncommitted state the invoking checkout happens to
    have: a fresh ``git worktree add --detach`` checkout is always clean at
    creation, so the test's "compatible receipt reuses" assertion does not
    depend on this session's own working tree being clean.
    """
    lane = tmp_path / "lane"
    head = _git(ROOT, "rev-parse", "HEAD").stdout.strip()
    _git(ROOT, "worktree", "add", "--detach", str(lane), head)
    try:
        yield lane, head
    finally:
        subprocess.run(
            ["git", "-C", str(ROOT), "worktree", "remove", "--force", str(lane)],
            capture_output=True,
            text=True,
            timeout=60,
        )


def test_pre_push_hook_reuses_receipt_after_routing_from_a_foreign_interpreter(
    lane_worktree: tuple[Path, str],
) -> None:
    lane, head = lane_worktree
    real_venv = ROOT / ".venv"
    assert real_venv.is_dir(), "this checkout must already have a provisioned .venv"

    # The lane's own venv: `reexec_into_lane` looks for exactly this path.
    # Symlinking the whole `.venv` directory (not just `bin/python`) is what
    # keeps `pyvenv.cfg`-driven site-packages resolution -- and therefore
    # every third-party dependency `devtools` needs -- intact.
    lane_venv = lane / ".venv"
    lane_venv.symlink_to(real_venv)
    lane_python = lane_venv / "bin" / "python"

    # A second, "inherited coordinator" venv at an unrelated path -- same
    # underlying interpreter and packages (symlinked the same way), but its
    # OWN `.venv` parent is NOT `lane`, so `checkout_guard._python_environment_root`
    # reports it as belonging to a different checkout when sampled from `lane`.
    foreign_root = tmp_path_sibling(lane, "inherited-coordinator")
    foreign_venv = foreign_root / ".venv"
    foreign_venv.symlink_to(real_venv)
    foreign_python = foreign_venv / "bin" / "python"
    # `reexec_into_lane` compares invoked paths, not resolved targets -- these
    # two symlinks must differ by path even though they resolve to the same
    # underlying interpreter, or the mismatch this test exists to exercise
    # never triggers.
    assert str(foreign_python.absolute()) != str(lane_python.absolute())

    # Reference environment fingerprint: exactly what a correctly-routed
    # `devtools verify --quick` would have recorded for this lane, computed
    # by running the SAME command `reexec_into_lane` would re-exec into.
    fingerprint_script = (
        "import json\n"
        "from pathlib import Path\n"
        "from devtools.checkout_guard import assert_polylogue_matches_checkout\n"
        f"fp = assert_polylogue_matches_checkout(Path({str(lane)!r}), context='pre-push')\n"
        "print(json.dumps(fp.as_dict()))\n"
    )
    reference = subprocess.run(
        [str(lane_python), "-c", fingerprint_script],
        cwd=lane,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert reference.returncode == 0, reference.stdout + reference.stderr
    environment = json.loads(reference.stdout)

    fingerprint = worktree_fingerprint(lane)
    receipt_path = lane / ".cache" / "verify" / "current-run.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(
            {
                "tier": "quick",
                "status": "success",
                "exit_code": 0,
                "git_head": head,
                "final_git_head": head,
                "checkout_root": str(lane.resolve()),
                "worktree_fingerprint": fingerprint,
                "final_worktree_fingerprint": fingerprint,
                "environment_fingerprint": environment,
            }
        ),
        encoding="utf-8",
    )

    updates_file = lane.parent / "updates.txt"
    updates_file.write_text(f"refs/heads/topic {head} refs/heads/topic {head}\n", encoding="utf-8")

    # The real hook route, invoked exactly as `.beads-hooks/pre-push` does
    # (`python -m devtools.pre_push_gate <updates-file>`), but under the
    # "foreign" interpreter and with cwd at the lane root -- the inherited-
    # coordinator-environment shape the fix must route around.
    result = subprocess.run(
        [str(foreign_python), "-m", "devtools.pre_push_gate", str(updates_file)],
        cwd=lane,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "pre-push: compatible quick receipt reused" in result.stderr, result.stdout + result.stderr
    assert "pre-push: running quick verification baseline" not in result.stderr


def tmp_path_sibling(anchor: Path, name: str) -> Path:
    sibling = anchor.parent / name
    sibling.mkdir(parents=True, exist_ok=True)
    return sibling
