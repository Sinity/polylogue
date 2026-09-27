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

A cache counts as warm only when it carries a completion stamp keyed to the
effective inputs (mypy's version and the passthrough arguments), written after
a run that left a complete cache; a partial, interrupted or differently
configured cache is cold. Each checkout's own seed-check-publish lifecycle is
serialized by a checkout-local lock, so two gates in one checkout never write
the same cache at once.

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
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import tomllib

#: mypy exits 0 (clean) or 1 (type errors) with a complete cache; anything
#: else is a crash or a usage error and may leave a partial one.
_CACHE_COMPLETE_EXITS = frozenset({0, 1})
#: Written inside a cache after a run that left it complete.
_STAMP = ".polylogue-complete"
#: Passthrough options that disable or redirect the module cache. A run using
#: one produces no reusable cache here, so it is not managed: it runs
#: serialized on the shared lock and neither seeds, stamps nor publishes.
_UNMANAGED_OPTIONS = ("--no-incremental", "--cache-dir")


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


def _input_key(root: Path, mypy_args: list[str]) -> str:
    """Digest of what decides whether a cache can be reused.

    mypy's version, the passthrough arguments and the ``[tool.mypy]``
    configuration: a sibling that narrowed ``files`` publishes a partial cache
    that a full-corpus checkout must not accept as warm.
    """
    try:
        version = importlib.metadata.version("mypy")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    try:
        with (root / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle).get("tool", {}).get("mypy", {})
    except (OSError, tomllib.TOMLDecodeError):
        config = None
    payload = json.dumps([version, mypy_args, config], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _unmanaged(mypy_args: list[str]) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in mypy_args for option in _UNMANAGED_OPTIONS)


def _remove_abandoned(directory: Path, prefix: str) -> None:
    """Remove staging copies a killed or cancelled gate left behind; caller holds the lock."""
    for stale in directory.glob(f"{prefix}*"):
        shutil.rmtree(stale, ignore_errors=True)


def _is_complete(path: Path, key: str) -> bool:
    try:
        return (path / _STAMP).read_text(encoding="utf-8") == key
    except OSError:
        return False


def _copy_tree(source: Path, target: Path) -> None:
    """Copy *source* to the absent *target*, sharing extents where the filesystem can.

    Timestamps are preserved: mypy's filesystem cache rejects an entry whose
    data file's mtime differs from the one its metadata recorded.
    """
    completed = subprocess.run(
        ["cp", "-R", "--reflink=auto", "--preserve=timestamps", "--", str(source), str(target)],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target)


def _seed(local: Path, shared: Path) -> None:
    """Give *local* a copy of the shared cache; caller holds the lock."""
    staging = local.with_name(f"{local.name}.seed-{os.getpid()}")
    _remove_abandoned(local.parent, f"{local.name}.seed-")
    local.parent.mkdir(parents=True, exist_ok=True)
    _copy_tree(shared, staging)
    shutil.rmtree(local, ignore_errors=True)
    os.replace(staging, local)


def _publish(local: Path, shared: Path) -> None:
    """Replace the shared cache with *local*'s; caller holds the lock."""
    staging = shared.with_name(f"{shared.name}.publish-{os.getpid()}")
    retired = shared.with_name(f"{shared.name}.retired-{os.getpid()}")
    _remove_abandoned(shared.parent, f"{shared.name}.publish-")
    _remove_abandoned(shared.parent, f"{shared.name}.retired-")
    _copy_tree(local, staging)
    if shared.exists():
        os.replace(shared, retired)
    os.replace(staging, shared)
    shutil.rmtree(retired, ignore_errors=True)


def _check(mypy: Path, cache: Path, root: Path, mypy_args: list[str], key: str) -> int:
    """Run mypy on *cache*, stamping it complete only after a run that left it so."""
    cache.mkdir(parents=True, exist_ok=True)
    (cache / _STAMP).unlink(missing_ok=True)
    returncode = subprocess.run([str(mypy), "--cache-dir", str(cache), *mypy_args], cwd=root, check=False).returncode
    if returncode in _CACHE_COMPLETE_EXITS:
        (cache / _STAMP).write_text(key, encoding="utf-8")
    return returncode


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

    if _unmanaged(mypy_args):
        with _locked(lock_path):
            return subprocess.run([str(mypy), *mypy_args], cwd=root, check=False).returncode

    key = _input_key(root, mypy_args)
    local.parent.mkdir(parents=True, exist_ok=True)
    # Lock order is always checkout, then shared, so the two cannot deadlock.
    with _locked(local.parent / "mypy.lock"):
        if not _is_complete(local, key):
            with _locked(lock_path):
                if _is_complete(shared, key):
                    _seed(local, shared)
                else:
                    # No sibling has a usable cache: this is the one cold
                    # scan. It runs under the shared lock so concurrent
                    # siblings wait and seed from its result instead of
                    # scanning cold beside it.
                    returncode = _check(mypy, local, root, mypy_args, key)
                    if returncode in _CACHE_COMPLETE_EXITS:
                        _publish(local, shared)
                    return returncode

        returncode = _check(mypy, local, root, mypy_args, key)
        if returncode in _CACHE_COMPLETE_EXITS:
            with _locked(lock_path):
                _publish(local, shared)
        return returncode


if __name__ == "__main__":
    raise SystemExit(main())
