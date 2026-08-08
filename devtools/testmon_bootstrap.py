"""Bootstrap a fresh worktree's pytest-testmon cache from the main checkout.

The hazard this closes (polylogue-mq4vx): every fresh agent worktree lane
re-seeds pytest-testmon from scratch. `devtools verify --seed-testmon` seeds
the affected-selection dependency database with a full non-integration pytest
run; that run costs real wall-clock (the main checkout's
``.cache/testmon/testmondata`` is ~28MB, built from the whole non-integration
suite). A linked worktree with no local seed either pays that cost again or
hits the unseeded-refusal preflight in ``devtools/verify.py``
(``_testmon_preflight``) and blocks entirely.

But the seed database is copyable. ``pytest-testmon`` records ``file_fp``
entries keyed by path **relative to the invoking repo root**, each with a
per-file content checksum (``fsha``). A worktree is a distinct working tree
sharing the same relative layout as the main checkout, so a testmondata file
copied verbatim from main is immediately meaningful there: any file that
differs between the worktree and the main checkout at copy time
self-invalidates (its ``fsha`` won't match), and testmon correctly treats the
tests that depend on it as affected on the very next run. No merge or rewrite
is needed for the relative file fingerprints.

The reusable stamp is typed. It records collection completeness, graph
completeness, baseline color, and whether the graph is exact or rebound to a
new checkout. A red graph is allowed for affected selection only. Bootstrap
revalidates the SQLite graph after the online backup and recomputes its file
fingerprint because SQLite backup can produce a byte-different equivalent
database.

This module owns exactly one decision and one action:

- :func:`decide_testmon_bootstrap` -- pure decision, no subprocess beyond the
  caller. It validates the main stamp or a complete red seed attempt.
- :func:`bootstrap_testmon_seed_files` -- the copy action once bootstrapping
  has been decided.
- :func:`maybe_bootstrap_testmon_seed` -- the orchestrator `devtools verify`
  calls: detects whether ``repo_root`` is a linked worktree (via
  ``git rev-parse --absolute-git-dir --git-common-dir``, the same mechanism
  ``devtools/verify_worktree.py`` uses), finds the main checkout, and wires
  the decision to the action.

Concurrency: the main checkout may be mid-seed (a live ``--seed-testmon`` run
appending to its own testmondata) at the exact moment a worktree bootstraps
from it. ``testmondata`` is a real sqlite database, so a naive byte copy of an
open, actively-written file can capture a torn, inconsistent snapshot. This
copies it through :meth:`sqlite3.Connection.backup`, sqlite's own online-backup
API -- built for copying a live database without an exclusive lock, immune to
concurrent writers by design. ``seed.json`` is a small file written atomically
by ``verify.py`` (write-temp-then-rename), so bootstrap writes its newly bound
stamp atomically after the copied graph has been revalidated.

This module NEVER writes to the main checkout's copy of either file --
only reads from main, only writes to ``repo_root``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

from devtools.testmon_state import (
    TestmonSeedStamp,
    refresh_stamp,
    stamp_from_attempt,
    validate_stamp,
)

TESTMON_DATA_RELPATH = ".cache/testmon/testmondata"
TESTMON_SEED_STAMP_RELPATH = ".cache/testmon/seed.json"
TESTMON_SEED_ATTEMPT_RELPATH = ".cache/testmon/seed-attempt.json"


@dataclass(frozen=True)
class BootstrapDecision:
    """Whether a worktree's testmon cache should be bootstrapped from main, and why."""

    should_bootstrap: bool
    reason: str
    main_testmon_data: Path | None = None
    main_seed_stamp: Path | None = None
    main_seed_attempt: Path | None = None
    main_checkout_root: Path | None = None
    protocol_version: int = 4


def _checkout_root_for_data(data_path: Path) -> Path:
    """Resolve the checkout root for canonical and test-local cache layouts."""
    resolved = data_path.resolve()
    if resolved.parent.name == "testmon" and resolved.parent.parent.name == ".cache":
        return resolved.parents[2]
    return resolved.parent


def _is_valid_complete_seed_stamp(
    seed_stamp: Path,
    testmon_data: Path,
    *,
    protocol_version: int,
    checkout_root: Path,
) -> bool:
    """Validate both the typed stamp and the real SQLite graph it describes."""
    return (
        validate_stamp(
            seed_stamp,
            testmon_data,
            checkout_root=checkout_root,
            protocol_version=protocol_version,
        )
        is not None
    )


def decide_testmon_bootstrap(
    *,
    is_linked_worktree: bool,
    local_testmon_data: Path,
    local_seed_stamp: Path,
    main_testmon_data: Path,
    main_seed_stamp: Path,
    protocol_version: int,
    main_seed_attempt: Path | None = None,
    main_checkout_root: Path | None = None,
    local_checkout_root: Path | None = None,
) -> BootstrapDecision:
    """Decide whether to copy the main checkout's testmon seed into a worktree.

    Pure with respect to process state (no subprocess, no git): every input is
    an already-resolved path or flag, so this is directly unit-testable with
    tmp dirs standing in for "local worktree" and "main checkout".
    """
    if not is_linked_worktree:
        return BootstrapDecision(False, "repo_root is not a linked worktree; nothing to bootstrap")
    local_root = (local_checkout_root or _checkout_root_for_data(local_testmon_data)).resolve()
    if (
        local_testmon_data.is_file()
        and local_seed_stamp.is_file()
        and _is_valid_complete_seed_stamp(
            local_seed_stamp,
            local_testmon_data,
            protocol_version=protocol_version,
            checkout_root=local_root,
        )
    ):
        return BootstrapDecision(False, "local .cache/testmon already has a validated testmondata + seed stamp")
    if not main_testmon_data.is_file():
        return BootstrapDecision(
            False,
            "main checkout has no valid testmon graph because its testmondata file is missing",
        )
    root = main_checkout_root or _checkout_root_for_data(main_testmon_data)
    root = root.resolve()
    try:
        main_testmon_data.resolve().relative_to(root)
        main_seed_stamp.resolve().relative_to(root)
    except ValueError:
        return BootstrapDecision(False, "main testmon paths are not bound to the declared checkout root")
    if _is_valid_complete_seed_stamp(
        main_seed_stamp,
        main_testmon_data,
        protocol_version=protocol_version,
        checkout_root=root,
    ):
        return BootstrapDecision(
            True,
            f"main checkout has a validated testmon graph ({main_seed_stamp}); bootstrapping worktree cache",
            main_testmon_data=main_testmon_data,
            main_seed_stamp=main_seed_stamp,
            main_checkout_root=root,
            protocol_version=protocol_version,
        )
    if main_seed_attempt is not None and main_seed_attempt.is_file():
        try:
            attempt = json.loads(main_seed_attempt.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            attempt = None
        if (
            isinstance(attempt, dict)
            and stamp_from_attempt(
                attempt,
                main_testmon_data,
                checkout_root=root,
                protocol_version=protocol_version,
            )
            is not None
        ):
            return BootstrapDecision(
                True,
                "main checkout has a validated complete graph from a red seed attempt; bootstrapping worktree cache",
                main_testmon_data=main_testmon_data,
                main_seed_attempt=main_seed_attempt,
                main_checkout_root=root,
                protocol_version=protocol_version,
            )
    if main_seed_stamp.is_file():
        return BootstrapDecision(False, "main checkout seed stamp is stale, malformed, or graph-incomplete")
    return BootstrapDecision(
        False,
        "main checkout has no validated reusable testmon state",
    )


def _atomic_write_stamp(seed_stamp: Path, stamp: TestmonSeedStamp) -> None:
    seed_stamp.parent.mkdir(parents=True, exist_ok=True)
    tmp = seed_stamp.with_name(f"{seed_stamp.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(stamp.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(seed_stamp)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_copy_sqlite_db(src: Path, dst: Path) -> None:
    """Copy a (possibly concurrently-written) sqlite db via the online backup API.

    `sqlite3.Connection.backup` is designed to snapshot a live database without
    requiring an exclusive lock on the source, so this tolerates the main
    checkout mid-write. The destination is built at a temp path and only
    `rename`d into place once the backup completes, so a reader never observes
    a partially-copied file at `dst`.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)
    try:
        src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        try:
            dst_conn = sqlite3.connect(tmp)
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()
        finally:
            src_conn.close()
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


def bootstrap_testmon_seed_files(
    decision: BootstrapDecision,
    *,
    local_testmon_data: Path,
    local_seed_stamp: Path,
    checkout_root: Path | None = None,
    inherited_from: Path | None = None,
) -> bool:
    """Perform the copy `decision` describes and report whether it was stamped."""
    if not decision.should_bootstrap:
        return True
    assert decision.main_testmon_data is not None
    if decision.main_seed_stamp is None and decision.main_seed_attempt is None:
        return False
    if checkout_root is None or inherited_from is None:
        return False
    stamp: TestmonSeedStamp | None = None
    try:
        source_root = (decision.main_checkout_root or inherited_from).resolve()
        destination_root = checkout_root.resolve()
        if source_root == destination_root:
            return False
        if inherited_from.resolve() != source_root:
            return False
        if decision.main_testmon_data.resolve() == local_testmon_data.resolve():
            return False
        decision.main_testmon_data.resolve().relative_to(source_root)
        local_testmon_data.resolve().relative_to(destination_root)
        if decision.main_seed_stamp is not None:
            stamp = validate_stamp(
                decision.main_seed_stamp,
                decision.main_testmon_data,
                checkout_root=source_root,
                protocol_version=decision.protocol_version,
            )
        else:
            assert decision.main_seed_attempt is not None
            source = json.loads(decision.main_seed_attempt.read_text(encoding="utf-8"))
            if not isinstance(source, dict):
                return False
            stamp = stamp_from_attempt(
                source,
                decision.main_testmon_data,
                checkout_root=source_root,
                protocol_version=decision.protocol_version,
            )
            if stamp is None:
                return False
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError, sqlite3.Error):
        return False
    if stamp is None:
        return False
    try:
        _atomic_copy_sqlite_db(decision.main_testmon_data, local_testmon_data)
        stamp = stamp.rebound(checkout_root=destination_root, inherited_from=source_root)
        refreshed = refresh_stamp(stamp, local_testmon_data)
        if refreshed is None or refreshed.graph != stamp.graph:
            return False
        _atomic_write_stamp(local_seed_stamp, refreshed)
        return True
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return False


def _git_worktree_info(repo_root: Path) -> tuple[bool, Path] | None:
    """Return `(is_linked_worktree, main_checkout_path)`, or None if undeterminable.

    Same `git rev-parse --absolute-git-dir --git-common-dir` mechanism
    `devtools/verify_worktree.py:inspect_worktree` uses: a linked worktree's
    git-dir (`.git/worktrees/<name>`) differs from the shared
    git-common-dir; a main checkout's git-dir *is* the common-dir.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--absolute-git-dir", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    if len(lines) < 2:
        return None
    git_dir = Path(lines[0]).resolve()
    raw_common = Path(lines[1])
    common_dir = raw_common.resolve() if raw_common.is_absolute() else (repo_root / raw_common).resolve()
    is_linked = git_dir != common_dir
    main_checkout = common_dir.parent
    return is_linked, main_checkout


def maybe_bootstrap_testmon_seed(
    repo_root: Path,
    *,
    testmon_data_relpath: str = TESTMON_DATA_RELPATH,
    seed_stamp_relpath: str = TESTMON_SEED_STAMP_RELPATH,
    protocol_version: int,
) -> str | None:
    """Bootstrap `repo_root`'s testmon seed from its main checkout if warranted.

    Returns a one-line message to log on success, or ``None`` when no
    bootstrap happened (not a linked worktree, already seeded locally, or the
    main checkout has nothing valid to offer). Called from
    `devtools/verify.py` before `_testmon_preflight`, so a freshly-bootstrapped
    worktree passes that preflight instead of refusing.
    """
    info = _git_worktree_info(repo_root)
    if info is None:
        return None
    is_linked_worktree, main_checkout = info
    if main_checkout == repo_root.resolve():
        return None
    local_testmon_data = repo_root / testmon_data_relpath
    local_seed_stamp = repo_root / seed_stamp_relpath
    main_testmon_data = main_checkout / testmon_data_relpath
    main_seed_stamp = main_checkout / seed_stamp_relpath
    main_seed_attempt = main_checkout / TESTMON_SEED_ATTEMPT_RELPATH
    decision = decide_testmon_bootstrap(
        is_linked_worktree=is_linked_worktree,
        local_testmon_data=local_testmon_data,
        local_seed_stamp=local_seed_stamp,
        main_testmon_data=main_testmon_data,
        main_seed_stamp=main_seed_stamp,
        protocol_version=protocol_version,
        main_seed_attempt=main_seed_attempt,
        main_checkout_root=main_checkout,
        local_checkout_root=repo_root,
    )
    if not decision.should_bootstrap:
        return None
    stamped = bootstrap_testmon_seed_files(
        decision,
        local_testmon_data=local_testmon_data,
        local_seed_stamp=local_seed_stamp,
        checkout_root=repo_root,
        inherited_from=main_checkout,
    )
    if not stamped:
        return (
            f"verify: bootstrapped pytest-testmon seed into {local_testmon_data.parent}, "
            "but could not record its checkout provenance"
        )
    return (
        f"verify: bootstrapped pytest-testmon seed from main checkout {main_checkout} "
        f"into {local_testmon_data.parent} (worktree had no local seed)"
    )


__all__ = [
    "TESTMON_DATA_RELPATH",
    "TESTMON_SEED_STAMP_RELPATH",
    "TESTMON_SEED_ATTEMPT_RELPATH",
    "BootstrapDecision",
    "decide_testmon_bootstrap",
    "bootstrap_testmon_seed_files",
    "maybe_bootstrap_testmon_seed",
]
