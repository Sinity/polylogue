"""Owned, measured scratch leases for managed pytest invocations.

Pytest deliberately retains a supplied ``--basetemp`` tree. The managed test
runners therefore give every invocation a private lease and reclaim only that
lease after recording its terminal footprint. A killed runner leaves a marker
behind; a later invocation may reclaim it only after proving its owner process
is gone.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from devtools.cloud_sentinels import cloud_sentinel_declined

DEFAULT_SCRATCH_ROOT = Path("/realm/tmp/polylogue-pytest")
FALLBACK_SCRATCH_ROOT = Path("/tmp/polylogue-pytest")
_SUBPROCESS_RUN = subprocess.run
_LEASE_FILE = "lease.json"
_MAX_STALE_LEASES_PER_START = 32
_MAX_FAILURE_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_FAILURE_ARTIFACT_FILE_BYTES = 8 * 1024 * 1024
_MAX_FAILURE_ARTIFACT_FILES = 1_000


@dataclass(frozen=True)
class ScratchUsage:
    apparent_bytes: int
    allocated_bytes: int
    file_count: int
    directory_count: int


def measure_tree(root: Path) -> ScratchUsage:
    """Count a tree without following symlinks or failing a test teardown."""
    apparent = allocated = files = directories = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = tuple(os.scandir(directory))
        except OSError:
            continue
        directories += 1
        for entry in entries:
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if entry.is_dir(follow_symlinks=False):
                pending.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                files += 1
                apparent += info.st_size
                allocated += info.st_blocks * 512
    return ScratchUsage(apparent, allocated, files, directories)


def _size_bytes(raw: str) -> int | None:
    match = re.fullmatch(r"([0-9.]+)(B|KiB|MiB|GiB|TiB)", raw)
    if match is None:
        return None
    scale = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}[match.group(2)]
    return int(float(match.group(1)) * scale)


def btrfs_extent_usage(root: Path) -> dict[str, int] | None:
    """Return Btrfs exclusive/shared bytes when the local tool can prove them."""
    try:
        result = _SUBPROCESS_RUN(
            ["btrfs", "filesystem", "du", "-s", str(root)], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[0] == "Total":
            continue
        total, exclusive, shared = (_size_bytes(field) for field in fields[:3])
        if total is not None and exclusive is not None and shared is not None:
            return {
                "btrfs_total_bytes": total,
                "btrfs_exclusive_bytes": exclusive,
                "btrfs_shared_bytes": shared,
            }
    return None


def scratch_root_from_environment(environment: Mapping[str, str] | None = None) -> Path:
    resolved_environment = os.environ if environment is None else environment
    raw = resolved_environment.get("POLYLOGUE_PYTEST_BASETEMP_ROOT")
    if raw and not cloud_sentinel_declined("POLYLOGUE_PYTEST_BASETEMP_ROOT", raw):
        return Path(raw).expanduser()
    return DEFAULT_SCRATCH_ROOT if DEFAULT_SCRATCH_ROOT.parent.is_dir() else FALLBACK_SCRATCH_ROOT


def _process_start_ticks(pid: int) -> str | None:
    try:
        # Field 22 is the start time. Split after the only parenthesized field,
        # whose command text may contain spaces.
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _owner_is_alive(payload: dict[str, Any]) -> bool:
    pid = payload.get("pid")
    start = payload.get("process_start_ticks")
    return isinstance(pid, int) and isinstance(start, str) and _process_start_ticks(pid) == start


def _read_lease(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _prune_stale_leases(root: Path) -> tuple[str, ...]:
    removed: list[str] = []
    leases_root = root / "runs"
    if not leases_root.is_dir():
        return ()
    for marker in sorted(leases_root.glob(f"*/*/{_LEASE_FILE}"))[:_MAX_STALE_LEASES_PER_START]:
        payload = _read_lease(marker)
        if payload is None or not _owner_is_alive(payload):
            lease_root = marker.parent
            shutil.rmtree(lease_root, ignore_errors=True)
            removed.append(str(lease_root))
    # A completed lane removes its leaf. Reclaim empty run containers later;
    # active or sibling lanes keep their container non-empty and untouched.
    for run_root in sorted(leases_root.iterdir()):
        with contextlib.suppress(OSError):
            run_root.rmdir()
            removed.append(str(run_root))
    return tuple(removed)


def _capture_failure_artifacts(source: Path, destination: Path) -> dict[str, Any]:
    """Keep small diagnostic files, never a second incident-scale corpus."""
    copied: list[str] = []
    skipped: list[dict[str, Any]] = []
    total = 0
    candidates = sorted(path for path in source.rglob("*") if path.is_file() and not path.is_symlink())
    for path in candidates[:_MAX_FAILURE_ARTIFACT_FILES]:
        relative = path.relative_to(source)
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > _MAX_FAILURE_ARTIFACT_FILE_BYTES or total + size > _MAX_FAILURE_ARTIFACT_BYTES:
            skipped.append({"path": str(relative), "size": size, "reason": "artifact_budget"})
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            shutil.copy2(path, target, follow_symlinks=False)
            total += size
            copied.append(str(relative))
    payload = {"copied_bytes": total, "copied_files": copied, "skipped": skipped}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


Outcome = Literal["success", "failure", "cancelled", "worker_crash"]


@dataclass
class PytestScratchLease:
    root: Path
    lease_root: Path
    basetemp: Path
    evidence_dir: Path
    run_id: str
    lane: str
    stale_leases_reclaimed: tuple[str, ...]

    @classmethod
    def acquire(cls, *, root: Path, run_id: str, lane: str, evidence_dir: Path) -> PytestScratchLease:
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        stale = _prune_stale_leases(root)
        safe_lane = re.sub(r"[^a-z0-9_.-]+", "-", lane.lower()).strip("-") or "pytest"
        lease_root = root / "runs" / run_id / safe_lane
        lease_root.mkdir(parents=True, exist_ok=False)
        marker = {
            "run_id": run_id,
            "lane": safe_lane,
            "pid": os.getpid(),
            "process_start_ticks": _process_start_ticks(os.getpid()),
            "created_monotonic_ns": time.monotonic_ns(),
        }
        (lease_root / _LEASE_FILE).write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
        return cls(root, lease_root, lease_root / "pytest", evidence_dir, run_id, safe_lane, stale)

    def command(self, command: list[str]) -> list[str]:
        if any(argument == "--basetemp" or argument.startswith("--basetemp=") for argument in command):
            raise ValueError("managed pytest owns --basetemp; do not supply a shared path")
        return [*command, f"--basetemp={self.basetemp}"]

    def environment(self, environment: dict[str, str]) -> dict[str, str]:
        return {
            **environment,
            "POLYLOGUE_PYTEST_SCRATCH_ROOT": str(self.basetemp),
            "POLYLOGUE_PYTEST_SCRATCH_LANE": self.lane,
        }

    def finalize(self, outcome: Outcome) -> dict[str, Any]:
        terminal = measure_tree(self.lease_root)
        extent = btrfs_extent_usage(self.lease_root)
        artifacts: dict[str, Any] | None = None
        if outcome != "success" and self.basetemp.exists():
            artifacts = _capture_failure_artifacts(self.basetemp, self.evidence_dir / "scratch-failure-artifacts")
        cleanup_started = time.monotonic_ns()
        shutil.rmtree(self.lease_root, ignore_errors=True)
        # A verifier run can own several sequential lanes. Remove only this
        # run's now-empty container, never the shared ``runs`` namespace or a
        # sibling lane still in flight.
        with contextlib.suppress(OSError):
            self.lease_root.parent.rmdir()
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "lane": self.lane,
            "outcome": outcome,
            "terminal_usage": asdict(terminal),
            "high_water_usage": asdict(terminal),
            "stale_leases_reclaimed": list(self.stale_leases_reclaimed),
            "cleanup_complete": not self.lease_root.exists(),
            "cleanup_started_monotonic_ns": cleanup_started,
        }
        if extent is not None:
            payload["extent_usage"] = extent
        if artifacts is not None:
            payload["failure_artifacts"] = artifacts
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        (self.evidence_dir / "scratch-metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return payload


__all__ = ["PytestScratchLease", "ScratchUsage", "btrfs_extent_usage", "measure_tree", "scratch_root_from_environment"]
