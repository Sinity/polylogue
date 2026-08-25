"""Owned, measured scratch leases for managed pytest invocations.

Pytest deliberately retains a supplied ``--basetemp`` tree. The managed test
runners therefore give every invocation a private lease and reclaim only that
lease after recording its terminal footprint. A killed runner leaves a marker
behind; a later invocation may reclaim it only after proving its owner process
is gone.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from devtools.cloud_sentinels import cloud_sentinel_declined, running_in_cloud_sandbox

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
    if DEFAULT_SCRATCH_ROOT.parent.is_dir():
        return DEFAULT_SCRATCH_ROOT
    if running_in_cloud_sandbox():
        return FALLBACK_SCRATCH_ROOT
    raise RuntimeError("managed pytest requires the workstation scratch mount at /realm/tmp")


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


@contextlib.contextmanager
def _lease_lock(root: Path) -> Any:
    """Serialize lease creation and stale cleanup within one scratch root."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lease.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_lease(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _remove_owned_tree(path: Path) -> None:
    """Remove an authenticated lease even when tests hardened its fixtures."""
    for directory, child_directories, _files in os.walk(path, topdown=True, followlinks=False):
        os.chmod(directory, stat.S_IRWXU, follow_symlinks=False)
        for child in child_directories:
            candidate = Path(directory, child)
            if not candidate.is_symlink():
                os.chmod(candidate, stat.S_IRWXU, follow_symlinks=False)
    shutil.rmtree(path)


def _prune_stale_leases(root: Path) -> tuple[str, ...]:
    removed: list[str] = []
    leases_root = root / "runs"
    if not leases_root.is_dir():
        return ()
    for marker in sorted(leases_root.glob(f"*/*/{_LEASE_FILE}"))[:_MAX_STALE_LEASES_PER_START]:
        payload = _read_lease(marker)
        if payload is None or not _owner_is_alive(payload):
            lease_root = marker.parent
            _remove_owned_tree(lease_root)
            removed.append(str(lease_root))
    # A completed lane removes its leaf. Reclaim empty run containers later;
    # active or sibling lanes keep their container non-empty and untouched.
    for run_root in sorted(leases_root.iterdir()):
        with contextlib.suppress(OSError):
            run_root.rmdir()
            removed.append(str(run_root))
    return tuple(removed)


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


def _open_directory_chain(root: Path, parts: tuple[str, ...], *, create: bool = False) -> int:
    fd = os.open(root, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
    try:
        for part in parts:
            if create:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(part, mode=0o700, dir_fd=fd)
            next_fd = os.open(part, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _copy_bounded_file(source: Path, destination: Path, relative: Path, limit: int) -> tuple[str, int] | None:
    """Copy one regular file through no-following directory descriptors."""
    source_fd = _open_directory_chain(source, tuple(relative.parts[:-1]))
    try:
        destination_fd = _open_directory_chain(destination, tuple(relative.parts[:-1]), create=True)
        try:
            target_name = relative.name
            target_fd: int | None = None
            source_file_fd = os.open(target_name, os.O_RDONLY | _NOFOLLOW, dir_fd=source_fd)
            try:
                source_stat = os.fstat(source_file_fd)
                if not stat.S_ISREG(source_stat.st_mode):
                    return None
                size = source_stat.st_size
                if size > _MAX_FAILURE_ARTIFACT_FILE_BYTES:
                    return "artifact_budget", size
                if size > limit:
                    return "artifact_budget", size
                target_fd = os.open(
                    target_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
                    mode=0o600,
                    dir_fd=destination_fd,
                )
                remaining = size
                while remaining:
                    chunk = os.read(source_file_fd, min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    view = memoryview(chunk)
                    while view:
                        written = os.write(target_fd, view)
                        view = view[written:]
                    remaining -= len(chunk)
                if remaining:
                    os.close(target_fd)
                    target_fd = None
                    os.unlink(target_name, dir_fd=destination_fd)
                    return None
                return "copied", size
            finally:
                os.close(source_file_fd)
                if target_fd is not None:
                    os.close(target_fd)
        finally:
            os.close(destination_fd)
    except OSError:
        return None
    finally:
        os.close(source_fd)


def _iter_failure_files(source: Path) -> Any:
    """Yield a bounded prefix without materializing or sorting the whole tree."""
    pending = [source]
    scanned = 0
    scan_limit = _MAX_FAILURE_ARTIFACT_FILES * 16
    while pending and scanned < scan_limit:
        directory = pending.pop()
        try:
            entries = os.scandir(directory)
        except OSError:
            continue
        with entries:
            for entry in entries:
                scanned += 1
                if scanned > scan_limit:
                    break
                try:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        yield Path(entry.path)
                except OSError:
                    continue


def _capture_failure_artifacts(source: Path, destination: Path) -> dict[str, Any]:
    """Keep small diagnostic files, never a second incident-scale corpus."""
    copied: list[str] = []
    skipped: list[dict[str, Any]] = []
    total = 0
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError:
        return {"copied_bytes": 0, "copied_files": [], "skipped": [], "error": "destination_not_owned"}
    for path in _iter_failure_files(source):
        if len(copied) + len(skipped) >= _MAX_FAILURE_ARTIFACT_FILES:
            break
        relative = path.relative_to(source)
        result = _copy_bounded_file(source, destination, relative, _MAX_FAILURE_ARTIFACT_BYTES - total)
        if result is None:
            continue
        status, size = result
        if status == "artifact_budget":
            skipped.append({"path": str(relative), "size": size, "reason": status})
            continue
        total += size
        copied.append(str(relative))
    payload = {"copied_bytes": total, "copied_files": copied, "skipped": skipped}
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
    lease_token: str

    @classmethod
    def acquire(cls, *, root: Path, run_id: str, lane: str, evidence_dir: Path) -> PytestScratchLease:
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        with _lease_lock(root):
            stale = _prune_stale_leases(root)
            safe_lane = re.sub(r"[^a-z0-9_.-]+", "-", lane.lower()).strip("-") or "pytest"
            run_root = root / "runs" / run_id
            run_root.mkdir(parents=True, exist_ok=True)
            lease_root = run_root / safe_lane
            staging_root = run_root / f".{safe_lane}.creating-{os.getpid()}-{uuid4().hex}"
            staging_root.mkdir(parents=False)
            lease_token = uuid4().hex
            marker = {
                "run_id": run_id,
                "lane": safe_lane,
                "lease_token": lease_token,
                "pid": os.getpid(),
                "process_start_ticks": _process_start_ticks(os.getpid()),
                "created_monotonic_ns": time.monotonic_ns(),
            }
            (staging_root / _LEASE_FILE).write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
            staging_root.rename(lease_root)
        return cls(root, lease_root, lease_root / "pytest", evidence_dir, run_id, safe_lane, stale, lease_token)

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
        with _lease_lock(self.root):
            marker = _read_lease(self.lease_root / _LEASE_FILE)
            owns_lease = isinstance(marker, dict) and marker.get("lease_token") == self.lease_token
            terminal = measure_tree(self.lease_root) if owns_lease else ScratchUsage(0, 0, 0, 0)
            high_water = _event_high_water(self.evidence_dir, terminal)
            extent = btrfs_extent_usage(self.lease_root) if owns_lease else None
            artifacts: dict[str, Any] | None = None
            if owns_lease and outcome != "success" and self.basetemp.exists():
                artifacts = _capture_failure_artifacts(self.basetemp, self.evidence_dir / "scratch-failure-artifacts")
            cleanup_started = time.monotonic_ns()
            if owns_lease:
                _remove_owned_tree(self.lease_root)
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
            "high_water_usage": asdict(high_water),
            "high_water_scope": "observed_test_trees_and_terminal_lease",
            "high_water_complete": False,
            "owner_verified": owns_lease,
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


def _event_high_water(evidence_dir: Path, terminal: ScratchUsage) -> ScratchUsage:
    """Combine per-test and per-worker observations without inventing values."""
    high_water = asdict(terminal)
    event_paths = [evidence_dir / "events.jsonl"]
    with contextlib.suppress(OSError):
        event_paths.extend((evidence_dir / "events").glob("*.jsonl"))
    for event_path in event_paths:
        with contextlib.suppress(OSError):
            for line in event_path.read_text(encoding="utf-8", errors="replace").splitlines():
                with contextlib.suppress(json.JSONDecodeError):
                    payload = json.loads(line)
                    observed = payload.get("high_water") if isinstance(payload, dict) else None
                    if isinstance(observed, dict):
                        for key in high_water:
                            value = observed.get(key)
                            if isinstance(value, int):
                                high_water[key] = max(high_water[key], value)
    return ScratchUsage(**high_water)


def run_managed_pytest(
    command: list[str], *, cwd: Path, env: Mapping[str, str], stdout: Any = None, stderr: Any = None
) -> subprocess.CompletedProcess[Any]:
    """Run pytest in an owned process group and reap it before lease cleanup."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=dict(env),
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    try:
        process.wait()
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    return subprocess.CompletedProcess(command, process.returncode)


__all__ = [
    "PytestScratchLease",
    "ScratchUsage",
    "btrfs_extent_usage",
    "measure_tree",
    "run_managed_pytest",
    "scratch_root_from_environment",
]
