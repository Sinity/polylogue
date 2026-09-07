"""Serialize concurrent mypy gates and share their incremental cache."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import subprocess
import sys
from pathlib import Path


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args, mypy_args = parser.parse_known_args(argv)
    root = args.root.resolve()
    shared = _git_common_dir(root) / "polylogue-mypy"
    shared.mkdir(parents=True, exist_ok=True)
    lock_path = shared / "lock"
    cache_path = shared / "cache"
    cache_path.mkdir(exist_ok=True)
    mypy = root / ".venv" / "bin" / "mypy"
    if not mypy.is_file():
        print(f"mypy gate: missing {mypy}", file=sys.stderr)
        return 127
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            completed = subprocess.run(
                [str(mypy), "--cache-dir", str(cache_path), *mypy_args],
                cwd=root,
                check=False,
            )
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
