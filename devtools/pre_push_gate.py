"""Select the smallest safe verification gate for a Git pre-push update.

Git passes one update per line on stdin.  A push whose complete diff only
changes ``.beads/`` cannot affect Python, generated product surfaces, or the
type graph, so it runs the two Beads graph-integrity checks instead of the
full quick gate.  Mixed and ordinary code pushes retain the normal gate.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from devtools import repo_root


@dataclass(frozen=True, slots=True)
class PushUpdate:
    local_ref: str
    local_sha: str
    remote_ref: str
    remote_sha: str


def parse_updates(text: str) -> list[PushUpdate]:
    updates: list[PushUpdate] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"malformed pre-push update on line {number}: expected four fields")
        updates.append(PushUpdate(*parts))
    return updates


def _is_zero_sha(value: str) -> bool:
    return bool(value) and set(value) == {"0"}


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _base_for_new_ref(local_sha: str, *, cwd: Path) -> str:
    for candidate in ("origin/master", "origin/main", "master", "main"):
        exists = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", candidate],
            cwd=cwd,
            check=False,
            capture_output=True,
        )
        if exists.returncode == 0:
            merged = subprocess.run(
                ["git", "merge-base", local_sha, candidate],
                cwd=cwd,
                check=False,
                capture_output=True,
                text=True,
            )
            if merged.returncode == 0 and merged.stdout.strip():
                return merged.stdout.strip()
    return _git("hash-object", "-t", "tree", "/dev/null", cwd=cwd)


def changed_paths(updates: list[PushUpdate], *, cwd: Path) -> set[str]:
    paths: set[str] = set()
    for update in updates:
        if _is_zero_sha(update.local_sha):
            continue  # deleting a ref does not publish new file content
        base = _base_for_new_ref(update.local_sha, cwd=cwd) if _is_zero_sha(update.remote_sha) else update.remote_sha
        output = _git("diff", "--name-only", "--diff-filter=ACDMRTUXB", base, update.local_sha, cwd=cwd)
        paths.update(line for line in output.splitlines() if line)
    return paths


def is_beads_only(paths: set[str]) -> bool:
    return bool(paths) and all(path == ".beads" or path.startswith(".beads/") for path in paths)


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def run_gate(updates: list[PushUpdate], *, cwd: Path) -> str:
    paths = changed_paths(updates, cwd=cwd)
    if is_beads_only(paths):
        print("pre-push: Beads-only diff; checking JSONL and dependency graph.", file=sys.stderr)
        _run(
            [
                sys.executable,
                "-m",
                "devtools.verify_backlog_hygiene",
                "--checks",
                "D1,D2",
                ".beads/issues.jsonl",
            ],
            cwd=cwd,
        )
        return "beads"

    stamp = cwd / ".cache" / "last-verify-head"
    current = _git("rev-parse", "HEAD", cwd=cwd)
    if stamp.exists() and stamp.read_text(encoding="utf-8").splitlines()[:1] == [current]:
        print(f"pre-push: HEAD already verified ({current[:8]}); skipping.", file=sys.stderr)
        return "stamped"

    print("pre-push: running quick verification baseline", file=sys.stderr)
    _run([sys.executable, "-m", "devtools", "verify", "--quick"], cwd=cwd)
    return "quick"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("updates", type=Path, help="file containing Git pre-push update lines")
    args = parser.parse_args(argv)
    try:
        updates = parse_updates(args.updates.read_text(encoding="utf-8"))
        run_gate(updates, cwd=repo_root())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"pre-push: verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
