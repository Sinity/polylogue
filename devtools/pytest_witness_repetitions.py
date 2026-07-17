"""Bounded, receipt-bearing repetitions for the seed-hang witnesses.

The July seed failures were not ordinary unit-test failures: their defining
property was intermittent lifecycle loss under an xdist controller.  A single
green invocation cannot establish that those paths are repaired.  This module
therefore invokes the *ordinary* ``devtools test`` route once per witness and
mode, retaining every managed-run receipt even when an attempt times out or
fails.  It deliberately has no retry path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from devtools import repo_root

_CACHE_ROOT = Path(".cache") / "pytest-witness-repetitions"
_WORKER_RE = re.compile(r"\[(gw\d+)\]")
_OWNER_SHUTDOWN_GRACE_S = 5.0


@dataclass(frozen=True, slots=True)
class Witness:
    """One exact lifecycle witness and the event its real-route test awaits."""

    name: str
    nodeid: str
    awaited_lifecycle: str


WITNESSES: tuple[Witness, ...] = (
    Witness(
        name="periodic-wal-checkpoint",
        nodeid="tests/unit/daemon/test_daemon_cli.py::test_periodic_wal_checkpoint_targets_archive_root_tiers",
        awaited_lifecycle="daemon write coordinator invokes maintenance.wal_checkpoint then cancellation propagates",
    ),
    Witness(
        name="periodic-db-optimize",
        nodeid="tests/unit/daemon/test_daemon_cli.py::test_periodic_db_optimize_targets_archive_root_tiers",
        awaited_lifecycle="daemon write coordinator invokes maintenance.db_optimize then cancellation propagates",
    ),
    Witness(
        name="periodic-embedding-backlog",
        nodeid="tests/unit/daemon/test_embedding_convergence_progress.py::test_periodic_embedding_backlog_waits_for_catch_up_complete",
        awaited_lifecycle="catch_up_complete event gates the first maintenance.embedding_backlog drain",
    ),
)


@dataclass(frozen=True, slots=True)
class AttemptReceipt:
    """Durable result of one invocation; failures remain first-class evidence."""

    witness: str
    nodeid: str
    mode: str
    ordinal: int
    workers: int
    command: tuple[str, ...]
    started_at: str
    finished_at: str
    status: str
    exit_code: int | None
    duration_s: float | None
    node_duration_s: float | None
    worker_id: str | None
    archive_root_scope: str | None
    archive_root_cleaned: bool | None
    containment_receipt: str | None
    process_group_cleaned: bool | None
    awaited_lifecycle: str
    failure: str | None


@dataclass(frozen=True, slots=True)
class RepetitionReceipt:
    """One complete proof batch, including every passed and failed attempt."""

    format: str
    source_root: str
    git_head: str | None
    attempts_per_mode: int
    xdist_workers: int
    timeout_s: float
    started_at: str
    finished_at: str
    attempts: tuple[AttemptReceipt, ...]
    ok: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


Runner = Callable[[Sequence[str], Path, dict[str, str], float], subprocess.CompletedProcess[str]]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _git_head(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _run_direct(
    command: Sequence[str], root: Path, env: dict[str, str], timeout_s: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), cwd=root, env=env, text=True, capture_output=True, timeout=timeout_s, check=False
    )


def _run_directories(root: Path) -> set[Path]:
    runs = root / ".cache" / "verify" / "runs"
    return {path.parent for path in runs.glob("*/run.json")}


def _single_new_run(root: Path, before: set[Path]) -> Path | None:
    new = _run_directories(root) - before
    if len(new) != 1:
        return None
    return next(iter(new))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _node_metadata(report: dict[str, Any] | None, nodeid: str) -> tuple[float | None, str | None]:
    if report is None:
        return None, None
    tests = report.get("tests")
    if not isinstance(tests, list):
        return None, None
    for test in tests:
        if not isinstance(test, dict) or test.get("nodeid") != nodeid:
            continue
        duration = 0.0
        seen_duration = False
        worker_id: str | None = None
        for phase in ("setup", "call", "teardown"):
            value = test.get(phase)
            if not isinstance(value, dict):
                continue
            phase_duration = value.get("duration")
            if isinstance(phase_duration, (int, float)):
                duration += float(phase_duration)
                seen_duration = True
            longrepr = value.get("longrepr")
            if isinstance(longrepr, str):
                match = _WORKER_RE.search(longrepr)
                if match:
                    worker_id = match.group(1)
        return (duration if seen_duration else None), worker_id
    return None, None


def _receipt_from_run(
    run_dir: Path | None, *, root: Path, nodeid: str
) -> tuple[float | None, float | None, str | None, str | None, bool | None, str | None, bool | None]:
    if run_dir is None:
        return None, None, None, None, None, None, None
    run = _read_json(run_dir / "run.json")
    if run is None:
        return None, None, None, None, None, None, None
    steps = run.get("steps")
    step = steps[0] if isinstance(steps, list) and steps and isinstance(steps[0], dict) else {}
    report_path = step.get("report_path")
    report = _read_json(root / report_path) if isinstance(report_path, str) else None
    node_duration, worker_id = _node_metadata(report, nodeid)
    containment_path = run_dir / "steps" / "01-pytest-focused" / "containment.json"
    containment = _read_json(containment_path)
    if containment is None:
        current_path = step.get("containment_path")
        containment = _read_json(root / current_path) if isinstance(current_path, str) else None
    archive_root_scope = containment.get("tmpfs_cleanup_path") if containment else None
    if not isinstance(archive_root_scope, str):
        archive_root_scope = None
    archive_root_cleaned = not Path(archive_root_scope).exists() if archive_root_scope else None
    process_group_cleaned = containment.get("controller_group_alive") is False if containment else None
    return (
        float(step["duration_s"]) if isinstance(step.get("duration_s"), (int, float)) else None,
        node_duration,
        worker_id,
        archive_root_scope,
        archive_root_cleaned,
        str(containment_path),
        process_group_cleaned,
    )


def _await_timeout_cleanup(
    *, root: Path, before: set[Path], nodeid: str
) -> tuple[
    float | None,
    float | None,
    str | None,
    str | None,
    bool | None,
    str | None,
    bool | None,
]:
    """Wait briefly for the externally-owned supervisor to publish teardown.

    A subprocess timeout terminates the devtools owner, while the independent
    supervisor still needs a bounded interval to notice that death, kill the
    pytest group, clean its tmpfs root, and write the final receipt.  Reading
    before that transition would report a completed cleanup as a leak.
    """
    details = _receipt_from_run(None, root=root, nodeid=nodeid)
    for _ in range(100):
        run_dir = _single_new_run(root, before)
        details = _receipt_from_run(run_dir, root=root, nodeid=nodeid)
        if details[4] is True and details[6] is True:
            return details
        time.sleep(0.05)
    return details


def _command(*, nodeid: str, workers: int) -> list[str]:
    return [sys.executable, "-m", "devtools", "test", nodeid, "-n", str(workers)]


def _attempt(
    *,
    witness: Witness,
    mode: str,
    ordinal: int,
    workers: int,
    root: Path,
    timeout_s: float,
    runner: Runner,
) -> AttemptReceipt:
    command = _command(nodeid=witness.nodeid, workers=workers)
    before = _run_directories(root)
    started_at = _utc_now()
    env = os.environ.copy()
    env["POLYLOGUE_PYTEST_WORKERS"] = str(workers)
    try:
        completed = runner(command, root, env, timeout_s + _OWNER_SHUTDOWN_GRACE_S)
        completed_run_dir = _single_new_run(root, before)
        duration_s, node_duration_s, worker_id, archive_scope, archive_cleaned, containment, group_cleaned = (
            _receipt_from_run(completed_run_dir, root=root, nodeid=witness.nodeid)
        )
        failed = completed.returncode != 0
        failure = (completed.stderr or completed.stdout).strip()[-4000:] if failed else None
        return AttemptReceipt(
            witness=witness.name,
            nodeid=witness.nodeid,
            mode=mode,
            ordinal=ordinal,
            workers=workers,
            command=tuple(command),
            started_at=started_at,
            finished_at=_utc_now(),
            status="failed" if failed else "passed",
            exit_code=completed.returncode,
            duration_s=duration_s,
            node_duration_s=node_duration_s,
            worker_id=worker_id,
            archive_root_scope=archive_scope,
            archive_root_cleaned=archive_cleaned,
            containment_receipt=containment,
            process_group_cleaned=group_cleaned,
            awaited_lifecycle=witness.awaited_lifecycle,
            failure=failure,
        )
    except subprocess.TimeoutExpired as exc:
        # ``subprocess.run`` kills the devtools owner on timeout.  Its external
        # supervisor then owns the process-tree teardown; wait only for that
        # receipt to become observable and retain whatever it says.  Never run
        # the failed attempt again.
        duration_s, node_duration_s, worker_id, archive_scope, archive_cleaned, containment, group_cleaned = (
            _await_timeout_cleanup(root=root, before=before, nodeid=witness.nodeid)
        )
        return AttemptReceipt(
            witness=witness.name,
            nodeid=witness.nodeid,
            mode=mode,
            ordinal=ordinal,
            workers=workers,
            command=tuple(command),
            started_at=started_at,
            finished_at=_utc_now(),
            status="timed_out",
            exit_code=None,
            duration_s=duration_s,
            node_duration_s=node_duration_s,
            worker_id=worker_id,
            archive_root_scope=archive_scope,
            archive_root_cleaned=archive_cleaned,
            containment_receipt=containment,
            process_group_cleaned=group_cleaned,
            awaited_lifecycle=witness.awaited_lifecycle,
            failure=(
                f"managed invocation did not finish within {timeout_s + _OWNER_SHUTDOWN_GRACE_S:g}s "
                f"(including {timeout_s:g}s evidence bound): {exc}"
            ),
        )


def run_repetitions(
    *,
    source_root: Path | None = None,
    attempts_per_mode: int = 10,
    xdist_workers: int = 3,
    timeout_s: float = 10.0,
    runner: Runner = _run_direct,
) -> RepetitionReceipt:
    """Run every current witness in isolated and xdist mode without retries."""
    if attempts_per_mode < 1:
        raise ValueError("attempts_per_mode must be positive")
    if xdist_workers < 1:
        raise ValueError("xdist_workers must be positive")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    root = (source_root or repo_root()).resolve()
    started_at = _utc_now()
    attempts: list[AttemptReceipt] = []
    for witness in WITNESSES:
        for mode, workers in (("isolated", 0), ("xdist", xdist_workers)):
            for ordinal in range(1, attempts_per_mode + 1):
                attempts.append(
                    _attempt(
                        witness=witness,
                        mode=mode,
                        ordinal=ordinal,
                        workers=workers,
                        root=root,
                        timeout_s=timeout_s,
                        runner=runner,
                    )
                )
    ok = all(
        attempt.status == "passed"
        and attempt.duration_s is not None
        and attempt.duration_s < timeout_s
        and attempt.node_duration_s is not None
        and attempt.node_duration_s < timeout_s
        and attempt.archive_root_cleaned is True
        and attempt.process_group_cleaned is True
        for attempt in attempts
    )
    return RepetitionReceipt(
        format="pytest-witness-repetitions-v1",
        source_root=str(root),
        git_head=_git_head(root),
        attempts_per_mode=attempts_per_mode,
        xdist_workers=xdist_workers,
        timeout_s=timeout_s,
        started_at=started_at,
        finished_at=_utc_now(),
        attempts=tuple(attempts),
        ok=ok,
    )


def _default_output(root: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return root / _CACHE_ROOT / f"{stamp}-receipt.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repeat exact seed-hang witnesses through managed pytest.")
    parser.add_argument("--attempts", type=int, default=10, help="Consecutive attempts per witness and mode.")
    parser.add_argument("--xdist-workers", type=int, default=3, help="Workers for each xdist attempt.")
    parser.add_argument("--timeout-s", type=float, default=10.0, help="Per-invocation and per-node bound.")
    parser.add_argument("--output", type=Path, help="Durable JSON receipt destination.")
    parser.add_argument("--json", action="store_true", help="Print the complete receipt as JSON.")
    args = parser.parse_args(argv)
    root = repo_root().resolve()
    receipt = run_repetitions(
        source_root=root,
        attempts_per_mode=args.attempts,
        xdist_workers=args.xdist_workers,
        timeout_s=args.timeout_s,
    )
    output = (args.output or _default_output(root)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt.to_dict(), indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(receipt.to_dict(), indent=2))
    else:
        print(f"pytest witness repetitions {'passed' if receipt.ok else 'failed'}: {output}")
    return 0 if receipt.ok else 1


__all__ = ["AttemptReceipt", "RepetitionReceipt", "WITNESSES", "Witness", "main", "run_repetitions"]


if __name__ == "__main__":
    raise SystemExit(main())
