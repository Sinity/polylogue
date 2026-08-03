"""lane-init: provision a fanout lane worktree that can actually verify itself.

High-concurrency fanouts (16+ parallel lanes) have one hard prerequisite this
command owns end to end: every lane worktree needs its OWN virtualenv, because
a worktree reusing the main checkout's venv resolves ``import polylogue`` to
the main checkout (editable-install ``.pth``), and ``devtools verify``/pytest
correctly refuse to run there (checkout guard, exit 125). Until now the fix
was a manual multi-minute ``nix develop`` per worktree or skipping lane-side
verification entirely -- the 2026-08-03 fanout evidence shows neither scales.

One invocation, from the coordinator's checkout:

  1. Creates the worktree + branch if missing (from ``--base``, default
     ``origin/master``); reuses them if already present.
  2. Provisions an isolated venv via ``uv sync`` (``--frozen`` when
     ``uv.lock`` exists). With a warm uv cache this is seconds, not minutes.
  3. Proves the guard invariant: the lane venv's ``import polylogue`` must
     resolve inside the lane worktree, else exit non-zero loudly.
  4. Runs the standard ``verify-worktree`` checks (linked, distinct,
     expected branch).
  5. Appends a lane record to the coordinator-side ledger
     ``.cache/fanout/lanes.jsonl`` (append-only JSONL: lane name, worktree,
     branch, base sha, bead ids, venv state, timestamp) -- the resumable
     fanout state polylogue-in94 asks for, seeded here so dispatch/recovery
     tooling has one place to look.
  6. Prints the recommended per-lane resource env for the requested
     concurrency (``POLYLOGUE_PYTEST_WORKERS``) so 16 lanes do not
     oversubscribe the host by default.

Usage:
    devtools workspace lane-init /realm/worktrees/lane-cursors \\
        --branch feature/sources/cursor-catchup --beads polylogue-2qrx,polylogue-ix5r
    devtools workspace lane-init /realm/worktrees/lane-x --branch feature/x --no-venv
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from devtools import repo_root

LEDGER_RELPATH = Path(".cache/fanout/lanes.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("worktree", help="lane worktree path (created if missing)")
    parser.add_argument("--branch", required=True, help="lane branch (created from --base if missing)")
    parser.add_argument("--base", default="origin/master", help="base ref for a new branch (default: origin/master)")
    parser.add_argument("--beads", default="", help="comma-separated bead ids this lane owns")
    parser.add_argument("--no-venv", action="store_true", help="skip venv provisioning (lane will NOT be able to run devtools/pytest)")
    parser.add_argument("--expected-lanes", type=int, default=16, help="planned concurrent lane count, sizes the per-lane worker hint (default: 16)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit the lane record as JSON on stdout")
    return parser


def recommended_workers(expected_lanes: int, cpu_count: int | None = None) -> int:
    """Per-lane pytest worker budget: fair CPU share, floor 1, cap 4."""
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 8)
    if expected_lanes < 1:
        expected_lanes = 1
    return max(1, min(4, cpus // expected_lanes))


def lane_record(
    *,
    worktree: Path,
    branch: str,
    base_sha: str,
    beads: Sequence[str],
    venv: bool,
    workers: int,
) -> dict[str, object]:
    return {
        "lane": worktree.name,
        "worktree": str(worktree),
        "branch": branch,
        "base_sha": base_sha,
        "beads": list(beads),
        "venv": venv,
        "pytest_workers": workers,
        "status": "provisioned",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def coordinator_root(start: Path) -> Path:
    """Resolve the MAIN checkout root (ledger home) even when run from a linked worktree."""
    probe = _run(["git", "-C", str(start), "rev-parse", "--path-format=absolute", "--git-common-dir"])
    if probe.returncode == 0:
        common = Path(probe.stdout.strip())
        if common.name == ".git":
            return common.parent
    return start


def append_ledger(root: Path, record: dict[str, object]) -> Path:
    path = root / LEDGER_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _run(cmd: Sequence[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), cwd=cwd, text=True, capture_output=True, check=False)


def _branch_exists(root: Path, branch: str) -> bool:
    return _run(["git", "-C", str(root), "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"]).returncode == 0


def _ensure_worktree(root: Path, worktree: Path, branch: str, base: str) -> str | None:
    """Create the worktree/branch if missing. Returns an error string or None."""
    if worktree.exists():
        probe = _run(["git", "-C", str(worktree), "rev-parse", "--show-toplevel"])
        if probe.returncode != 0:
            return f"{worktree} exists but is not a git worktree"
        return None
    if _branch_exists(root, branch):
        result = _run(["git", "-C", str(root), "worktree", "add", str(worktree), branch])
    else:
        result = _run(["git", "-C", str(root), "worktree", "add", "-b", branch, str(worktree), base])
    if result.returncode != 0:
        return f"git worktree add failed: {result.stderr.strip()}"
    return None


def _provision_venv(worktree: Path) -> str | None:
    # dev-common+speed = the devshell's effective test/verify surface without
    # platform-fragile extras (atheris has no cp314t wheel).
    cmd = ["uv", "sync", "--extra", "dev-common", "--extra", "speed"]
    if (worktree / "uv.lock").exists():
        cmd.append("--frozen")
    result = _run(cmd, cwd=worktree)
    if result.returncode != 0:
        return f"uv sync failed: {result.stderr.strip()[-800:]}"
    return None


def _guard_check(worktree: Path) -> str | None:
    python = worktree / ".venv" / "bin" / "python"
    if not python.exists():
        return f"no venv python at {python}"
    result = _run([str(python), "-c", "import polylogue, sys; print(polylogue.__file__)"])
    if result.returncode != 0:
        return f"lane venv cannot import polylogue: {result.stderr.strip()[-400:]}"
    resolved = result.stdout.strip()
    if not resolved.startswith(str(worktree) + os.sep):
        return (
            f"guard violation: lane venv resolves polylogue to {resolved!r}, "
            f"outside the lane worktree -- the shared-venv hijack this command exists to prevent"
        )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    worktree = Path(args.worktree).resolve()
    beads = [b.strip() for b in args.beads.split(",") if b.strip()]

    error = _ensure_worktree(root, worktree, args.branch, args.base)
    if error:
        print(f"lane-init: {error}", file=sys.stderr)
        return 1

    verify = _run(
        [sys.executable, "-m", "devtools", "workspace", "verify-worktree", str(worktree), "--expect-branch", args.branch],
        cwd=root,
    )
    if verify.returncode != 0:
        sys.stderr.write(verify.stdout + verify.stderr)
        print("lane-init: verify-worktree failed", file=sys.stderr)
        return 1

    if not args.no_venv:
        error = _provision_venv(worktree)
        if error:
            print(f"lane-init: {error}", file=sys.stderr)
            return 1
        error = _guard_check(worktree)
        if error:
            print(f"lane-init: {error}", file=sys.stderr)
            return 1

    base_sha = _run(["git", "-C", str(worktree), "rev-parse", "--short=9", "HEAD"]).stdout.strip()
    workers = recommended_workers(args.expected_lanes)
    record = lane_record(
        worktree=worktree,
        branch=args.branch,
        base_sha=base_sha,
        beads=beads,
        venv=not args.no_venv,
        workers=workers,
    )
    ledger_path = append_ledger(coordinator_root(root), record)

    if args.as_json:
        print(json.dumps(record, ensure_ascii=False, indent=2))
    else:
        print(f"lane ready: {worktree}")
        print(f"  branch: {args.branch} @ {base_sha}")
        print(f"  venv: {'provisioned (guard-verified)' if not args.no_venv else 'SKIPPED -- lane cannot run devtools/pytest'}")
        if beads:
            print(f"  beads: {', '.join(beads)}")
        print(f"  ledger: {ledger_path}")
        print(f"  dispatch env: POLYLOGUE_PYTEST_WORKERS={workers}  (for {args.expected_lanes} concurrent lanes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
