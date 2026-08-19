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
     ``uv.lock`` exists), pinned to the coordinator's own interpreter and
     verified to have actually been built from it. With a warm uv cache this
     is seconds, not minutes.
  3. Proves the guard invariant: the lane venv's ``import polylogue`` must
     resolve inside the lane worktree, else exit non-zero loudly.
  4. Runs the standard ``verify-worktree`` checks (linked, distinct,
     expected branch).
  5. Seeds the coordinator's testmon graph and reports whether it actually
     covers this lane's environment digest -- a lane whose first verify will
     bootstrap says so instead of claiming warmth.
  6. Appends a lane record to the coordinator-side ledger
     ``.cache/fanout/lanes.jsonl`` (append-only JSONL: lane name, worktree,
     branch, base sha, bead ids, venv state, timestamp) -- the resumable
     fanout state polylogue-in94 asks for, seeded here so dispatch/recovery
     tooling has one place to look.
  7. Prints the recommended per-lane resource env for the requested
     concurrency (``POLYLOGUE_PYTEST_WORKERS``) so 16 lanes do not
     oversubscribe the host by default.

Usage:
    devtools workspace lane-init /realm/worktrees/lane-cursors \\
        --branch feature/sources/cursor-catchup --beads polylogue-2qrx,polylogue-ix5r
    devtools workspace lane-init /realm/worktrees/lane-x --branch feature/x --no-venv
"""

from __future__ import annotations

import argparse
import contextlib
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worktree", help="lane worktree path (created if missing)")
    parser.add_argument("--branch", required=True, help="lane branch (created from --base if missing)")
    parser.add_argument("--base", default="origin/master", help="base ref for a new branch (default: origin/master)")
    parser.add_argument("--beads", default="", help="comma-separated bead ids this lane owns")
    parser.add_argument(
        "--no-venv", action="store_true", help="skip venv provisioning (lane will NOT be able to run devtools/pytest)"
    )
    parser.add_argument(
        "--expected-lanes",
        type=int,
        default=16,
        help="planned concurrent lane count, sizes the per-lane worker hint (default: 16)",
    )
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


def _run(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), cwd=cwd, env=env, text=True, capture_output=True, check=False)


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


def _base_executable_of(python: Path) -> Path | None:
    """Resolve the real interpreter behind a venv's ``python`` shim.

    ``sys._base_executable`` is the interpreter a venv was created FROM, which
    is what ``uv venv --python`` needs and what the testmon environment digest
    fingerprints. ``sys.executable`` inside a venv is the shim, not the build.
    """
    probe = _run([str(python), "-c", "import sys; print(sys._base_executable or sys.executable)"])
    if probe.returncode != 0:
        return None
    text = probe.stdout.strip()
    return Path(text) if text else None


def coordinator_base_interpreter(root: Path) -> Path | None:
    """The interpreter every lane must be built from: the coordinator's own.

    A lane provisioned with a DIFFERENT Python is not merely stylistically
    inconsistent, it is broken in two separate ways, both observed
    2026-08-19 (polylogue-l218h):

    1. ``uv`` left to its own resolution downloaded a free-threaded CPython
       3.14.5 for a lane whose coordinator runs Nix CPython 3.14.4. On that
       build ``import hypothesis`` raises ``AttributeError: 'installed_base'``
       from ``sysconfig``, so ``tests/conftest.py`` fails to import and EVERY
       ``devtools test`` invocation in the lane dies during collection. The
       lane looks provisioned and cannot run a single test.
    2. The interpreter is an input to the testmon environment digest, so a
       divergent one guarantees the lane's seeded graph can never match and
       its first verify pays a full bootstrap (~9.5x a warm run).

    Derived from the coordinator's venv rather than hardcoded: the store path
    changes on every nixpkgs bump, and a pinned literal would rot silently
    into the same class of mismatch it exists to prevent.
    """
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.exists():
        resolved = _base_executable_of(venv_python)
        if resolved is not None and resolved.exists():
            return resolved
    fallback = getattr(sys, "_base_executable", None) or sys.executable
    candidate = Path(fallback)
    return candidate if candidate.exists() else None


def _interpreter_guard(worktree: Path, expected: Path) -> str | None:
    """Prove the provisioned lane venv was actually built from ``expected``.

    ``uv sync --python`` can silently fall back (an already-present ``.venv``
    built from another interpreter is reused rather than rebuilt), so pinning
    the request is not the same as verifying the result.
    """
    python = worktree / ".venv" / "bin" / "python"
    if not python.exists():
        return f"no venv python at {python}"
    actual = _base_executable_of(python)
    if actual is None:
        return f"could not resolve the lane interpreter behind {python}"
    if actual.resolve() != expected.resolve():
        return (
            "lane interpreter does not match the coordinator's:\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            "a lane built from a different Python cannot share the coordinator's "
            "testmon graph, and has been observed unable to import hypothesis at all"
        )
    return None


#: Devshell-exported variables that describe the DEVSHELL's interpreter and
#: silently reroute a different interpreter's ``sysconfig``. A harness-spawned
#: lane inherits these, and the lane venv's Python then resolves build
#: configuration belonging to another build entirely -- pytest dies before
#: collecting anything, with ``AttributeError: 'installed_base'`` or
#: ``ModuleNotFoundError: No module named '_sysconfigdata_...'`` depending on
#: which pair of builds collide. Observed 2026-08-19 in a lane and in the
#: coordinator checkout (polylogue-l218h).
_INTERPRETER_DESCRIBING_ENV = (
    "_PYTHON_SYSCONFIGDATA_NAME",
    "_PYTHON_HOST_PLATFORM",
    "PYTHONPYCACHEPREFIX",
)


def _lane_env(worktree: Path) -> dict[str, str]:
    """Return an environment that cannot inherit another checkout's Python."""
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    # Not merely hygiene: these describe a specific interpreter BUILD, so
    # inheriting them into a venv built from a different one is fatal, and
    # PYTHONPYCACHEPREFIX additionally points bytecode caching at the
    # coordinator's .cache.
    for key in _INTERPRETER_DESCRIBING_ENV:
        env.pop(key, None)
    # uv's project-routing environment variables override subprocess.cwd.
    # Leaving either behind can install the coordinator checkout into this
    # lane's otherwise correctly selected .venv.
    env.pop("UV_PROJECT", None)
    env.pop("UV_WORKING_DIR", None)
    env["UV_PROJECT_ENVIRONMENT"] = str(worktree / ".venv")
    return env


def _provision_venv(worktree: Path, interpreter: Path | None = None) -> str | None:
    # dev-common+speed = the devshell's effective test/verify surface without
    # platform-fragile extras (atheris has no cp314t wheel).
    cmd = ["uv", "sync", "--extra", "dev-common", "--extra", "speed"]
    if interpreter is not None:
        # Without this uv picks an interpreter by its own resolution order and
        # will happily download one that does not match the coordinator.
        cmd.extend(["--python", str(interpreter)])
    if (worktree / "uv.lock").exists():
        cmd.append("--frozen")
    result = _run(cmd, cwd=worktree, env=_lane_env(worktree))
    if result.returncode != 0:
        return f"uv sync failed: {result.stderr.strip()[-800:]}"
    return None


def _guard_check(worktree: Path) -> str | None:
    python = worktree / ".venv" / "bin" / "python"
    if not python.exists():
        return f"no venv python at {python}"
    result = _run(
        [
            str(python),
            "-P",
            "-c",
            (
                "from pathlib import Path; "
                "from devtools.checkout_guard import assert_polylogue_matches_checkout; "
                "fingerprint = assert_polylogue_matches_checkout("
                "Path.cwd(), context='devtools workspace lane-init'); "
                "print(fingerprint.polylogue_import_path)"
            ),
        ],
        cwd=worktree,
        env=_lane_env(worktree),
    )
    if result.returncode != 0:
        return f"lane checkout guard failed: {result.stderr.strip()[-800:]}"
    return None


def lane_environment_digest(worktree: Path) -> str | None:
    """The testmon environment name this lane's own interpreter computes."""
    python = worktree / ".venv" / "bin" / "python"
    if not python.exists():
        return None
    probe = _run(
        [
            str(python),
            "-c",
            (
                "from pathlib import Path; "
                "from devtools.testmon_bootstrap import testmon_environment_digest; "
                "print(testmon_environment_digest(Path.cwd()))"
            ),
        ],
        cwd=worktree,
        env=_lane_env(worktree),
    )
    if probe.returncode != 0:
        return None
    return probe.stdout.strip() or None


def graph_environments(path: Path) -> set[str]:
    """Environment names a testmon graph actually holds, empty if unreadable."""
    import sqlite3

    if not path.is_file():
        return set()
    try:
        with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)) as conn:
            return {str(row[0]) for row in conn.execute("SELECT DISTINCT environment_name FROM environment")}
    except (OSError, sqlite3.Error):
        return set()


def _seed_testmon_graph(root: Path, worktree: Path) -> tuple[str, bool]:
    """Copy the coordinator's testmon graph into a lane, and say whether it helps.

    Returns ``(note, warm)``. ``warm`` is True only when the copied graph
    actually contains an environment matching the digest THIS LANE computes --
    which is the only thing that makes a lane's first verify an
    affected-selection run instead of a complete-corpus bootstrap.

    This used to report "seeded from main checkout (lane verifies start warm)"
    whenever the file copy succeeded, which is a claim about the wrong fact. A
    graph is useless to a lane unless it carries that lane's environment, and
    a copy can succeed while carrying none: the digest is invalidated by an
    interpreter change, a dependency bump, or an edit to ``tests/**/conftest.py``
    or ``devtools/pytest*.py``, so after any of those the coordinator's own
    graph goes stale for the coordinator too. Advertising warmth there sends a
    lane into a ~45-minute bootstrap it was told to expect in seconds
    (polylogue-l218h).

    A cold seed is reported, not fatal. A stale graph is the NORMAL state
    right after a dependency bump or conftest edit, and refusing to provision
    would break every lane for an expected condition; the honest note plus the
    ``testmon_warm`` field in the ledger record is what lets a coordinator
    decide to bootstrap once centrally instead of per lane.
    """
    from devtools.testmon_bootstrap import TESTMON_DATA_RELPATH

    source = root / TESTMON_DATA_RELPATH
    if not source.is_file():
        return "no coordinator graph to seed (first lane verify will bootstrap)", False
    destination = worktree / TESTMON_DATA_RELPATH
    import sqlite3

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("-wal", "-shm", "-journal"):
            Path(f"{destination}{suffix}").unlink(missing_ok=True)
        with (
            sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=30) as source_conn,
            sqlite3.connect(destination, timeout=30) as destination_conn,
        ):
            source_conn.backup(destination_conn)
    except (OSError, sqlite3.Error) as exc:
        return f"graph seed failed ({exc}); first lane verify will bootstrap", False

    environments = graph_environments(destination)
    if not environments:
        return "seeded graph holds no environments; first lane verify will bootstrap", False
    digest = lane_environment_digest(worktree)
    if digest is None:
        return (
            f"seeded {len(environments)} environment(s) but could not compute this lane's digest; warmth unverified"
        ), False
    if digest not in environments:
        return (
            f"seeded {len(environments)} environment(s), NONE matching this lane ({digest[:24]}...); "
            "first lane verify will bootstrap the complete corpus"
        ), False
    return f"seeded and matched this lane's environment ({digest[:24]}...); verifies start warm", True


def dispatch_env_lines(worktree: Path) -> tuple[str, ...]:
    """Environment a dispatched agent must apply before running lane tooling.

    ``_lane_env`` only sanitises the subprocesses lane-init itself spawns. An
    agent that later opens this worktree inherits the operator's or harness's
    environment untouched, so the same interpreter-describing variables that
    would have broken provisioning break its very first ``devtools test``
    instead. Printing the remedy next to the worker budget is the only place a
    dispatcher reliably reads.
    """
    lines = [f"export VIRTUAL_ENV={worktree / '.venv'}", 'export PATH="$VIRTUAL_ENV/bin:$PATH"']
    lines.extend(f"unset {key}" for key in _INTERPRETER_DESCRIBING_ENV)
    # gh / pr-scope reach GitHub over TLS; a devshell without this resolves no
    # CA bundle and every API call fails with a certificate error.
    lines.append("export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt")
    return tuple(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    worktree = Path(args.worktree).resolve()
    beads = [b.strip() for b in args.beads.split(",") if b.strip()]

    error = _ensure_worktree(root, worktree, args.branch, args.base)
    if error:
        print(f"lane-init: {error}", file=sys.stderr)
        return 1

    interpreter = None if args.no_venv else coordinator_base_interpreter(root)
    if not args.no_venv:
        error = _provision_venv(worktree, interpreter)
        if error:
            print(f"lane-init: {error}", file=sys.stderr)
            return 1
        if interpreter is not None:
            error = _interpreter_guard(worktree, interpreter)
            if error:
                print(f"lane-init: {error}", file=sys.stderr)
                return 1
        error = _guard_check(worktree)
        if error:
            print(f"lane-init: {error}", file=sys.stderr)
            return 1

    verify_python = worktree / ".venv" / "bin" / "python" if not args.no_venv else Path(sys.executable)
    verify = _run(
        [
            str(verify_python),
            "-m",
            "devtools",
            "workspace",
            "verify-worktree",
            str(worktree),
            "--expect-branch",
            args.branch,
        ],
        cwd=worktree if not args.no_venv else root,
        env=_lane_env(worktree) if not args.no_venv else None,
    )
    if verify.returncode != 0:
        sys.stderr.write(verify.stdout + verify.stderr)
        print("lane-init: verify-worktree failed", file=sys.stderr)
        return 1

    graph_note, graph_warm = _seed_testmon_graph(root, worktree)
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
    record["testmon_graph"] = graph_note
    record["testmon_warm"] = graph_warm
    record["interpreter"] = str(interpreter) if interpreter is not None else None
    record["dispatch_env"] = list(dispatch_env_lines(worktree))
    ledger_path = append_ledger(coordinator_root(root), record)

    if args.as_json:
        print(json.dumps(record, ensure_ascii=False, indent=2))
    else:
        print(f"lane ready: {worktree}")
        print(f"  branch: {args.branch} @ {base_sha}")
        print(
            f"  venv: {'provisioned (guard-verified)' if not args.no_venv else 'SKIPPED -- lane cannot run devtools/pytest'}"
        )
        if interpreter is not None:
            print(f"  interpreter: {interpreter} (verified)")
        print(f"  testmon graph: {graph_note}")
        if not graph_warm and not args.no_venv:
            print("  NOTE: this lane's first devtools verify will be a complete-corpus bootstrap.")
        if beads:
            print(f"  beads: {', '.join(beads)}")
        print(f"  ledger: {ledger_path}")
        print(f"  dispatch env: POLYLOGUE_PYTEST_WORKERS={workers}  (for {args.expected_lanes} concurrent lanes)")
        for line in dispatch_env_lines(worktree):
            print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
