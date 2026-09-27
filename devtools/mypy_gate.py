"""Run mypy on a checkout-local incremental cache seeded from its siblings.

Each checkout type-checks against its own cache, ``<root>/.cache/mypy``, so
sibling worktrees run side by side instead of queueing. A checkout without a
cache seeds one by reflink-copying the shared cache under the repository's
common git directory, and every run that leaves a valid cache publishes it
back there, so a new sibling starts warm rather than cold.

polylogue-r2lud serialized every checkout on one lock and one cache. It was
fixing concurrent *cold* scans: several full analyses at once, each over a
gigabyte, stalled for tens of minutes in I/O throttling. A seeded cache is
never cold. The one cold case left is a repository with no shared cache yet.
That run holds the lock while it checks, so concurrent siblings wait for it
and then seed from its result, and at most one cold scan runs.

The shared cache was also self-defeating. Sibling worktrees hold different
source, so each locked run rewrote the entries the previous sibling had just
written, and every run re-analysed the divergence while its siblings queued
behind it. Measured on 2026-09-27 at load average 16: a fresh seed's first
run took 50 s, a warm rerun 6.6 s, and four seeded runs side by side 10.7 s
wall-clock at about 420 MB RSS each. The same host's serialized quick gate
waited up to 107 s for the lock.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

#: mypy exits 0 (clean) or 1 (type errors) with a complete cache; anything
#: else is a crash or a usage error and may leave a partial one.
_CACHE_COMPLETE_EXITS = frozenset({0, 1})


def _git_common_dir(root: Path) -> Path:
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return root / ".cache"
    path = Path(common)
    return path if path.is_absolute() else (root / path).resolve()


@contextlib.contextmanager
def _locked(lock_path: Path) -> Iterator[None]:
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _has_cache(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _copy_tree(source: Path, target: Path) -> None:
    """Copy *source* to the absent *target*, sharing extents where the filesystem can."""
    completed = subprocess.run(
        ["cp", "-R", "--reflink=auto", "--", str(source), str(target)], capture_output=True, check=False
    )
    if completed.returncode != 0:
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target)


def _seed(local: Path, shared: Path) -> None:
    """Give *local* a copy of the shared cache; caller holds the lock."""
    staging = local.with_name(f"{local.name}.seed-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    local.parent.mkdir(parents=True, exist_ok=True)
    _copy_tree(shared, staging)
    shutil.rmtree(local, ignore_errors=True)
    os.replace(staging, local)


def _publish(local: Path, shared: Path) -> None:
    """Replace the shared cache with *local*'s; caller holds the lock."""
    staging = shared.with_name(f"{shared.name}.publish-{os.getpid()}")
    retired = shared.with_name(f"{shared.name}.retired-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    _copy_tree(local, staging)
    if shared.exists():
        os.replace(shared, retired)
    os.replace(staging, shared)
    shutil.rmtree(retired, ignore_errors=True)


def _run_mypy(mypy: Path, cache: Path, root: Path, mypy_args: list[str]) -> int:
    return subprocess.run([str(mypy), "--cache-dir", str(cache), *mypy_args], cwd=root, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args, mypy_args = parser.parse_known_args(argv)
    root = args.root.resolve()
    shared_root = _git_common_dir(root) / "polylogue-mypy"
    shared_root.mkdir(parents=True, exist_ok=True)
    lock_path = shared_root / "lock"
    shared = shared_root / "cache"
    local = root / ".cache" / "mypy"
    mypy = root / ".venv" / "bin" / "mypy"
    if not mypy.is_file():
        print(f"mypy gate: missing {mypy}", file=sys.stderr)
        return 127

    if not _has_cache(local):
        with _locked(lock_path):
            if _has_cache(shared):
                _seed(local, shared)
            else:
                # No sibling has a cache yet: this is the one cold scan. It
                # runs under the lock so concurrent siblings wait and seed
                # from its result instead of scanning cold beside it.
                local.mkdir(parents=True, exist_ok=True)
                returncode = _run_mypy(mypy, local, root, mypy_args)
                if returncode in _CACHE_COMPLETE_EXITS:
                    _publish(local, shared)
                return returncode

    returncode = _run_mypy(mypy, local, root, mypy_args)
    if returncode in _CACHE_COMPLETE_EXITS:
        with _locked(lock_path):
            _publish(local, shared)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
