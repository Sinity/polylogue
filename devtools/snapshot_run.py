"""Run a command against an immutable snapshot of the checkout.

The verification pipeline conflates two different needs. A receipt must name
what was actually tested (attestation integrity), and pytest-testmon's edges
must not be fingerprinted against content that changed underneath them (graph
correctness). Both were served by watching the working tree and, on any
mutation, aborting the run *and* unlinking the whole testmon database --
which made the repository's own workflow self-defeating, since ordinary bead
writes and agent activity mutate tracked files constantly.

Snapshot isolation serves both needs without forbidding anything. The command
runs against a frozen copy, so the working tree is simply free: edit, commit,
run `bd`, start another agent -- none of it can reach the running command, and
the receipt names content that provably could not change.

Mechanism, and why each piece is what it is:

* **reflink copy**, not a btrfs subvolume snapshot. The checkout is not a
  subvolume (its inode is not 256), so `btrfs subvolume snapshot` does not
  apply. `cp --reflink` is copy-on-write on this filesystem and measured
  ~110ms for the 3,182-file tree, so the snapshot is effectively free.
* **the same absolute path**, via a bind mount. A virtualenv is path-sensitive
  and :mod:`devtools.checkout_guard` deliberately refuses an interpreter that
  belongs to a different checkout, so running from a copy at a *different*
  path would either break imports or trip the guard. Binding the snapshot over
  the checkout's own path keeps both correct by construction.
* **bwrap, not systemd**. `BindPaths=`/`TemporaryFileSystem=` are service-only
  properties; the pytest supervisor uses `systemd-run --scope`, and a scope
  cannot carry them. bwrap composes *inside* the existing scope, so systemd
  keeps owning cgroups and resource limits while bwrap owns the mount view.

What is deliberately NOT frozen: `.venv` (the interpreter environment, not the
code under test), `.cache` (so a run's testmon graph and receipts persist past
teardown), `.git`, and the pytest basetemp root. These are bound back to the
live paths on top of the snapshot.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

__all__ = [
    "SnapshotUnavailableError",
    "bubblewrap_available",
    "materialize_snapshot",
    "snapshot_command",
]

#: Paths bound back to the live checkout on top of the frozen snapshot. These
#: are either not code under test (`.venv`), or state that must outlive the
#: run (`.cache` carries the testmon graph and the receipts the caller reads).
_LIVE_BINDS: tuple[str, ...] = (".venv", ".cache", ".git")


class SnapshotUnavailableError(RuntimeError):
    """Raised when this machine cannot provide an isolated snapshot run."""


def bubblewrap_available() -> bool:
    """Whether `bwrap` is present and usable unprivileged."""
    if shutil.which("bwrap") is None:
        return False
    probe = subprocess.run(
        ["bwrap", "--dev-bind", "/", "/", "--", "true"],
        capture_output=True,
        timeout=30,
        check=False,
    )
    return probe.returncode == 0


def materialize_snapshot(root: Path, destination: Path | None = None) -> Path:
    """Copy the checkout's evidence-bearing files into a reflinked snapshot.

    Includes tracked files plus non-ignored untracked ones -- the same content
    :func:`devtools.verify_runs.worktree_fingerprint` attests to. Untracked
    means a newly written test still runs; ignored means build output and caches
    stay out, which is what keeps the copy cheap.
    """
    target = destination or Path(tempfile.mkdtemp(prefix="polylogue-snapshot-", dir="/realm/tmp"))
    target.mkdir(parents=True, exist_ok=True)
    listings = (
        ["git", "ls-files", "-z"],
        ["git", "ls-files", "-z", "--others", "--exclude-standard"],
    )
    for listing in listings:
        found = subprocess.run(listing, cwd=root, capture_output=True, check=True)
        paths = [entry for entry in found.stdout.split(b"\0") if entry]
        if not paths:
            continue
        # `--reflink=auto` rather than `=always`: copy-on-write where the
        # filesystem supports it, a plain copy where it does not, so this stays
        # correct off btrfs instead of failing.
        copy = subprocess.run(
            ["xargs", "-0", "cp", "-a", "--reflink=auto", "--parents", "-t", str(target)],
            cwd=root,
            input=found.stdout,
            capture_output=True,
            check=False,
        )
        if copy.returncode != 0:
            raise SnapshotUnavailableError(f"snapshot copy failed: {copy.stderr.decode('utf-8', 'replace').strip()}")
    return target


def snapshot_command(
    command: Sequence[str],
    *,
    root: Path,
    snapshot: Path,
    extra_binds: Sequence[Path] = (),
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Return ``command`` wrapped so it sees ``snapshot`` at ``root``'s path."""
    del env
    argv = ["bwrap", "--dev-bind", "/", "/", "--bind", str(snapshot), str(root)]
    for relative in _LIVE_BINDS:
        live = root / relative
        if live.exists():
            argv += ["--bind", str(live), str(live)]
    for extra in extra_binds:
        if extra.exists():
            argv += ["--bind", str(extra), str(extra)]
    argv += ["--chdir", str(root), "--"]
    argv += list(command)
    return argv


def run_in_snapshot(
    command: Sequence[str],
    *,
    root: Path | None = None,
    extra_binds: Sequence[Path] = (),
    keep_snapshot: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    """Run ``command`` against a frozen snapshot of the checkout.

    Raises :class:`SnapshotUnavailableError` rather than silently running against the
    live tree: a caller that asked for isolation and did not get it would
    otherwise record a receipt whose central claim is false.
    """
    checkout = (root or Path.cwd()).resolve()
    if not bubblewrap_available():
        raise SnapshotUnavailableError("bwrap is unavailable, so the checkout cannot be frozen for this run")
    snapshot = materialize_snapshot(checkout)
    try:
        argv = snapshot_command(command, root=checkout, snapshot=snapshot, extra_binds=extra_binds)
        return subprocess.run(argv, check=False, capture_output=False)
    finally:
        if not keep_snapshot:
            shutil.rmtree(snapshot, ignore_errors=True)


def default_extra_binds() -> tuple[Path, ...]:
    """Live paths a pytest run needs beyond the checkout itself."""
    from devtools.verify_runs import DEFAULT_PYTEST_BASETEMP_ROOT

    candidates = [DEFAULT_PYTEST_BASETEMP_ROOT.parent, Path("/dev/shm")]
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        candidates.append(Path(runtime_dir))
    return tuple(path for path in candidates if path.exists())
