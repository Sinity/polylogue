"""Drive one fresh archive build through ``polylogued run`` and measure it.

The daemon is the production process, started exactly as the service starts
it, with three differences that are all isolation rather than shortcuts:

* ``HOME`` is the corpus's ``home/`` tree, so the typed default sources
  (``~/.claude/projects``, ``~/.codex/sessions``, ...) resolve to the corpus;
* every XDG root and ``POLYLOGUE_ARCHIVE_ROOT`` point into the run directory,
  so nothing touches an operator archive;
* structured events are written as JSON lines for the report.

The driver watches the archive read-only until the build is terminal
(promoted index generation, every admitted cursor complete, no pending raw
parse, no open convergence debt, every readiness surface ready), stops the
daemon with SIGINT, and folds the event log, the ops-tier batch ledger, its
own process-tree samples and an output fingerprint into one receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from devtools.fresh_build_bench.corpus import EXPORT_ORIGINS, change_stamp, load_manifest, verify_manifest

#: Every archive-readiness domain a finished build must report ready. An
#: unknown or newly unready domain keeps the build non-terminal.
REQUIRED_READINESS_DOMAINS: Final = frozenset(
    {
        "archive_sessions",
        "raw_artifacts",
        "search",
        "session_profiles",
        "threads",
        "tool_usage",
        "latency_profiles",
    }
)
_CLOCK_TICKS: Final = os.sysconf("SC_CLK_TCK")
_PAGE_SIZE: Final = os.sysconf("SC_PAGE_SIZE")
_WORK_PROGRESS_READ_CHUNK_BYTES: Final = 64 * 1024


@dataclass(frozen=True, slots=True)
class RunConfig:
    corpus: Path
    work: Path
    candidate: Path
    python: str
    label: str
    profile: bool = False
    profile_interval_s: float = 0.01
    stall_timeout_s: float = 900.0
    poll_s: float = 2.0
    stable_polls: int = 2
    fingerprint: bool = True
    extra_env: tuple[tuple[str, str], ...] = ()
    #: Asserted budgets, e.g. {"rss_peak_mib": 3072}; see ``report.BUDGETS``.
    budgets: tuple[tuple[str, float], ...] = ()


# ---------------------------------------------------------------------------
# process-tree sampling


def _children(pid: int) -> list[int]:
    result: list[int] = []
    try:
        tasks = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return result
    for task in tasks:
        try:
            with open(f"/proc/{pid}/task/{task}/children", encoding="ascii") as stream:
                result.extend(int(child) for child in stream.read().split())
        except OSError:
            continue
    return result


def _tree(pid: int) -> list[int]:
    pending, seen = [pid], []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.append(current)
        pending.extend(_children(current))
    return seen


def _proc_stat(pid: int) -> tuple[int, int, int] | None:
    """(cpu ticks, rss bytes, threads) for one process."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as stream:
            raw = stream.read()
    except OSError:
        return None
    tail = raw[raw.rfind(b")") + 2 :].split()
    cpu = int(tail[11]) + int(tail[12])
    threads = int(tail[17])
    rss = int(tail[21]) * _PAGE_SIZE
    return cpu, rss, threads


def _proc_rss_hwm(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return 0
    return 0


def _proc_io(pid: int) -> tuple[int, int] | None:
    """(read bytes, write bytes), or ``None`` when the counters are unreadable."""
    try:
        with open(f"/proc/{pid}/io", encoding="ascii") as stream:
            fields = dict(line.split(": ", 1) for line in stream.read().splitlines())
    except OSError:
        return None
    return int(fields.get("read_bytes", 0)), int(fields.get("write_bytes", 0))


class TreeSampler:
    """One-second samples of the daemon process tree: RSS, CPU and block I/O."""

    def __init__(self, pid: int, *, origin: float, interval_s: float = 0.25) -> None:
        self.pid = pid
        self.interval_s = interval_s
        self.samples: list[tuple[float, int, float, int, int, int]] = []
        self._stop = threading.Event()
        #: The driver's launch instant, so samples share the receipt's clock.
        self._started = origin
        #: Last cumulative (cpu ticks, read bytes, write bytes) per process
        #: ever seen: a parse worker that exits keeps its share of the totals.
        self._cumulative: dict[int, tuple[int, int, int]] = {}
        #: The daemon process's own resident high-water mark (VmHWM): exact
        #: for the main process whatever the sampling interval misses.
        self.daemon_rss_hwm_bytes = 0
        self._thread = threading.Thread(target=self._run, name="fresh-build-tree-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _sample(self) -> None:
        rss = threads = 0
        for pid in _tree(self.pid):
            stat = _proc_stat(pid)
            if stat is None:
                continue
            rss += stat[1]
            threads += stat[2]
            io = _proc_io(pid)
            if io is None:
                # The process exited between the two reads: its last
                # successful counters stand, not zeros.
                previous = self._cumulative.get(pid)
                io = (previous[1], previous[2]) if previous is not None else (0, 0)
            self._cumulative[pid] = (stat[0], io[0], io[1])
        self.daemon_rss_hwm_bytes = max(self.daemon_rss_hwm_bytes, _proc_rss_hwm(self.pid))
        cpu_ticks = sum(value[0] for value in self._cumulative.values())
        read_bytes = sum(value[1] for value in self._cumulative.values())
        write_bytes = sum(value[2] for value in self._cumulative.values())
        if rss:
            self.samples.append(
                (
                    round(time.monotonic() - self._started, 3),
                    rss,
                    cpu_ticks / _CLOCK_TICKS,
                    threads,
                    read_bytes,
                    write_bytes,
                )
            )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def finish(self) -> None:
        self._stop.set()
        self._thread.join()


# ---------------------------------------------------------------------------
# archive observation (read-only)


@dataclass(slots=True)
class Observation:
    t: float
    cursor_rows: int = 0
    cursor_complete: int = 0
    cursor_excluded: int = 0
    cursor_failing: int = 0
    cursor_deferred: int = 0
    raw_rows: int = 0
    #: Raw rows with neither a parse time nor a parse error. Reported, not
    #: gating: a retained non-session artifact (a workflow journal, a
    #: sidecar) is never parsed as a session, and the ``raw_artifacts``
    #: readiness domain is the daemon's own verdict on raw completeness.
    raw_pending: int = 0
    raw_failed: int = 0
    memberships_pending: int = 0
    open_debt: int = 0
    debt_by_stage: dict[str, int] = field(default_factory=dict)
    #: Open debt rows whose retry time is still in the future (in backoff),
    #: by stage: a backlog that is waiting, not being worked.
    debt_waiting_by_stage: dict[str, int] = field(default_factory=dict)
    #: Retry activity, independent of useful progress.
    debt_attempts: int = 0
    #: Summed cursor failures, and cursors waiting on a scheduled retry.
    cursor_failures: int = 0
    cursor_retry_waiting: int = 0
    frontier_retry_at: str | None = None
    convergence_next_run_at: float | None = None
    debt_next_retry_at: str | None = None
    cursor_next_retry_at: str | None = None
    useful_progress_at_s: float | None = None
    activity_at_s: float | None = None
    #: Advancing ``daemon.work.progress`` events seen so far: long work
    #: (preparing a large source) that changes no archive row until it
    #: publishes, but reports the messages and bytes it has processed.
    work_progress: int = 0
    promoted_index: str | None = None
    readiness: dict[str, bool] = field(default_factory=dict)
    error: str | None = None
    #: Whether a failed read may succeed on a later poll (a busy or locked
    #: database, a file swapped by a generation promotion). A schema the
    #: driver cannot read (an older candidate) fails the same way forever.
    error_retryable: bool = True

    @property
    def intake_complete(self) -> bool:
        return (
            self.cursor_rows > 0
            and self.cursor_complete + self.cursor_excluded == self.cursor_rows
            and self.cursor_failing == 0
            and self.cursor_deferred == 0
            and self.memberships_pending == 0
        )

    @property
    def readiness_complete(self) -> bool:
        return self.readiness.keys() >= REQUIRED_READINESS_DOMAINS and all(self.readiness.values())

    @property
    def terminal(self) -> bool:
        return (
            self.intake_complete and self.promoted_index is not None and self.open_debt == 0 and self.readiness_complete
        )


def _ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
    conn.execute("PRAGMA query_only=1")
    return conn


def _promoted_index(archive: Path) -> str | None:
    pointer = archive / "index.db"
    if not pointer.is_symlink():
        return None
    try:
        target = pointer.resolve(strict=True)
        target.relative_to((archive / ".index-generations").resolve(strict=True))
    except (OSError, ValueError):
        return None
    return str(target)


def observe(archive: Path, started: float, *, readiness_max_age_s: float | None = None) -> Observation:
    observation = Observation(t=round(time.monotonic() - started, 3))
    ops_path, source_path = archive / "ops.db", archive / "source.db"
    if not ops_path.exists() or not source_path.exists():
        return observation
    try:
        with closing(_ro(ops_path)) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "ingest_cursor" in tables:
                row = conn.execute(
                    """
                    SELECT COUNT(*),
                           COALESCE(SUM(excluded = 0 AND byte_offset >= stat_size AND failure_count = 0
                                        AND deferred_end_offset IS NULL), 0),
                           COALESCE(SUM(excluded = 1), 0),
                           COALESCE(SUM(excluded = 0 AND failure_count > 0), 0),
                           COALESCE(SUM(excluded = 0 AND deferred_end_offset IS NOT NULL), 0)
                    FROM ingest_cursor
                    """
                ).fetchone()
                (
                    observation.cursor_rows,
                    observation.cursor_complete,
                    observation.cursor_excluded,
                    observation.cursor_failing,
                    observation.cursor_deferred,
                ) = (int(value) for value in row)
                cursor_row = conn.execute(
                    "SELECT COALESCE(SUM(failure_count), 0),"
                    " COALESCE(SUM(next_retry_at IS NOT NULL AND next_retry_at > ?), 0) FROM ingest_cursor",
                    (datetime.now(UTC).isoformat(),),
                ).fetchone()
                observation.cursor_failures, observation.cursor_retry_waiting = (int(v) for v in cursor_row)
                observation.cursor_next_retry_at = conn.execute(
                    "SELECT MIN(next_retry_at) FROM ingest_cursor WHERE next_retry_at > ?",
                    (datetime.now(UTC).isoformat(),),
                ).fetchone()[0]
            if "convergence_debt" in tables:
                for stage, count in conn.execute("SELECT stage, COUNT(*) FROM convergence_debt GROUP BY stage"):
                    observation.debt_by_stage[str(stage)] = int(count)
                observation.open_debt = sum(observation.debt_by_stage.values())
                observation.debt_attempts = int(
                    conn.execute("SELECT COALESCE(SUM(attempts), 0) FROM convergence_debt").fetchone()[0]
                )
                observation.frontier_retry_at = conn.execute(
                    "SELECT MIN(next_retry_at) FROM convergence_debt WHERE stage='raw_frontier_inspection'"
                ).fetchone()[0]
                now_iso = datetime.now(UTC).isoformat()
                observation.debt_next_retry_at = conn.execute(
                    "SELECT MIN(next_retry_at) FROM convergence_debt WHERE next_retry_at > ?", (now_iso,)
                ).fetchone()[0]
                for stage, count in conn.execute(
                    "SELECT stage, COUNT(*) FROM convergence_debt "
                    "WHERE next_retry_at IS NOT NULL AND next_retry_at > ? GROUP BY stage",
                    (now_iso,),
                ):
                    observation.debt_waiting_by_stage[str(stage)] = int(count)
        with closing(_ro(source_path)) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "raw_sessions" in tables:
                row = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(parsed_at_ms IS NULL AND parse_error IS NULL), 0),"
                    " COALESCE(SUM(parse_error IS NOT NULL), 0) FROM raw_sessions"
                ).fetchone()
                observation.raw_rows, observation.raw_pending, observation.raw_failed = (int(v) for v in row)
            if "raw_session_memberships" in tables:
                # A membership still undecided, ambiguous, deferred or
                # quarantined has not settled its raw into the archive.
                observation.memberships_pending = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM raw_session_memberships WHERE decision IS NULL"
                        " OR decision IN ('ambiguous', 'deferred') OR revision_authority = 'quarantined'"
                    ).fetchone()[0]
                )
        observation.promoted_index = _promoted_index(archive)
        if observation.intake_complete and observation.promoted_index is not None and observation.open_debt == 0:
            now = time.monotonic()
            cached_at = _last_readiness.get("at")
            if (
                isinstance(cached_at, float)
                and now - cached_at < (_READINESS_POLL_S if readiness_max_age_s is None else readiness_max_age_s)
                and _last_readiness.get("archive") == str(archive)
            ):
                observation.readiness = dict(_last_readiness["readiness"])  # type: ignore[call-overload]
            else:
                from polylogue.storage.archive_readiness import archive_readiness_status

                readiness = archive_readiness_status(archive)
                surfaces = readiness.get("surfaces", {}) if readiness.get("checked") is True else {}
                observation.readiness = {
                    str(name): isinstance(surface, dict) and surface.get("ready") is True
                    for name, surface in surfaces.items()
                }
                _last_readiness.update(at=now, archive=str(archive), readiness=dict(observation.readiness))
    except (OSError, sqlite3.Error) as exc:
        observation.error = f"{type(exc).__name__}: {exc}"
        observation.error_retryable = _retryable_observation_error(exc)
    return observation


def _useful_progress(previous: Observation | None, current: Observation) -> bool:
    """Accepted material, reduced required work, or a new publication/readiness."""
    if previous is None:
        return False
    return (
        any(
            getattr(current, key) > getattr(previous, key)
            for key in ("cursor_rows", "cursor_complete", "raw_rows", "work_progress")
        )
        or current.raw_rows - current.raw_pending - current.raw_failed
        > previous.raw_rows - previous.raw_pending - previous.raw_failed
        or any(getattr(current, key) < getattr(previous, key) for key in ("memberships_pending", "open_debt"))
        or any(current.debt_by_stage.get(stage, 0) < count for stage, count in previous.debt_by_stage.items())
        or current.promoted_index is not None
        and current.promoted_index != previous.promoted_index
        or any(ready and not previous.readiness.get(domain, False) for domain, ready in current.readiness.items())
    )


class WorkProgressTail:
    """Count advancing ``daemon.work.progress`` events appended to the event log.

    Reads only the bytes appended since the previous call. Counter high-water
    marks are scoped by productive identity in a private SQLite file, so a
    retry that resets its local counters cannot masquerade as new work and a
    long run does not retain one Python object per source recipe.
    """

    def __init__(self, events: Path, *, state_root: Path | None = None) -> None:
        self._events = events
        self._offset = 0
        self._pending = b""
        self._state_directory = tempfile.TemporaryDirectory(prefix="polylogue-work-progress-", dir=state_root)
        self._connection = sqlite3.connect(Path(self._state_directory.name) / "high-water.sqlite3")
        self._connection.execute("PRAGMA cache_size = -256")
        self._connection.execute("PRAGMA journal_mode = OFF")
        self._connection.execute("PRAGMA synchronous = OFF")
        self._connection.execute(
            """CREATE TABLE productive_high_water (
                phase TEXT NOT NULL,
                productive_id TEXT NOT NULL,
                messages INTEGER NOT NULL,
                bytes INTEGER NOT NULL,
                PRIMARY KEY (phase, productive_id)
            ) WITHOUT ROWID"""
        )
        self.convergence_next_run_at: float | None = None
        self.advancing = 0
        self._closed = False

    def close(self) -> None:
        """Close and remove the private high-water store."""
        if self._closed:
            return
        self._closed = True
        self._connection.close()
        self._state_directory.cleanup()

    def _consume_line(self, line: bytes) -> None:
        if b"daemon.work.progress" not in line and b"daemon.periodic.scheduled" not in line:
            return
        try:
            event = json.loads(line)
        except ValueError:
            return
        if event.get("event") == "daemon.periodic.scheduled" and event.get("loop") == "convergence_check":
            value = event.get("next_run_at")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                self.convergence_next_run_at = float(value)
            return
        if event.get("event") != "daemon.work.progress":
            return
        phase = str(event.get("phase"))
        productive_id = event.get("productive_id")
        unit_id = event.get("unit_id")
        if not isinstance(productive_id, str) or not productive_id or not isinstance(unit_id, str) or not unit_id:
            return
        counters = (int(event.get("messages") or 0), int(event.get("bytes") or 0))
        row = self._connection.execute(
            "SELECT messages, bytes FROM productive_high_water WHERE phase = ? AND productive_id = ?",
            (phase, productive_id),
        ).fetchone()
        previous = (0, 0) if row is None else (int(row[0]), int(row[1]))
        if counters[0] >= previous[0] and counters[1] >= previous[1] and counters != previous:
            self._connection.execute(
                """INSERT INTO productive_high_water (phase, productive_id, messages, bytes)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (phase, productive_id) DO UPDATE SET
                    messages = excluded.messages,
                    bytes = excluded.bytes""",
                (phase, productive_id, counters[0], counters[1]),
            )
            self.advancing += 1

    def poll(self) -> int:
        if self._closed:
            return self.advancing
        try:
            with self._events.open("rb") as handle:
                handle.seek(self._offset)
                while chunk := handle.read(_WORK_PROGRESS_READ_CHUNK_BYTES):
                    self._offset += len(chunk)
                    lines = (self._pending + chunk).split(b"\n")
                    self._pending = lines.pop()
                    with self._connection:
                        for line in lines:
                            self._consume_line(line)
        except OSError:
            return self.advancing
        return self.advancing


def _retryable_observation_error(exc: BaseException) -> bool:
    if isinstance(exc, sqlite3.OperationalError):
        message = str(exc).lower()
        # ``locking protocol`` is SQLITE_PROTOCOL: a WAL reader lost its race
        # for a read lock to a concurrent checkpoint or restart, and the next
        # attempt normally succeeds.
        return any(token in message for token in ("locked", "busy", "unable to open", "disk i/o", "locking protocol"))
    # Corruption or a schema this driver cannot read never heals by polling.
    return isinstance(exc, OSError)


#: Seconds between readiness reads once everything else has settled: the
#: readiness census reads every raw row, so it is not taken every poll.
_READINESS_POLL_S = 60.0
_last_readiness: dict[str, object] = {}


# ---------------------------------------------------------------------------
# environment and candidate identity


def _git(candidate: Path, *args: str) -> str:
    # ``--no-optional-locks``: identifying a candidate must never take the
    # index lock another git process (or a later checkout) needs.
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(candidate), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _filesystem(path: Path) -> str:
    try:
        output = subprocess.run(
            ["stat", "-f", "-c", "%T", str(path)], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return output


def _backing_device(path: Path) -> str:
    """The mount source backing ``path`` (``/dev/nvme0n1p2``, ``server:/export``), else its device number."""
    try:
        device = os.stat(path).st_dev
    except OSError:
        return "unknown"
    major, minor = os.major(device), os.minor(device)
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as stream:
            for line in stream:
                fields = line.split()
                if len(fields) > 4 and fields[2] == f"{major}:{minor}" and " - " in line:
                    source = line.split(" - ", 1)[1].split()
                    if len(source) >= 2:
                        return source[1]
    except OSError:
        pass
    return f"{major}:{minor}"


def _meminfo_kib(key: str) -> int | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith(key + ":"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def _cpu_model() -> str:
    """The processor model name (``/proc/cpuinfo``), else the platform's own answer."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                key, _, value = line.partition(":")
                if key.strip() in {"model name", "Hardware", "cpu model"} and value.strip():
                    return value.strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment(config: RunConfig) -> dict[str, Any]:
    # Probed from the candidate, where the daemon is launched: a relative
    # ``--python`` names the same file for both.
    probe = json.loads(
        subprocess.run(
            [
                config.python,
                "-c",
                "import json,os,platform,sys;print(json.dumps([platform.python_version(),"
                " int(sys._is_gil_enabled()), sys.version, os.path.realpath(sys.executable)]))",
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=config.candidate,
        ).stdout
    )
    from polylogue.pipeline.parsed_tree_size import effective_physical_memory_bytes
    from polylogue.runtime import available_cpus

    return {
        # Same-sized workers on different processors are not the same host.
        "cpu_model": _cpu_model(),
        "host_cpu_count": os.cpu_count(),
        "host_mem_total_kib": _meminfo_kib("MemTotal"),
        # What the daemon (a child in this process's cgroup) sizes its worker
        # pools and memory budgets from: affinity and cgroup quotas included.
        "effective_cpus": available_cpus(),
        "effective_memory_bytes": effective_physical_memory_bytes(),
        "host_mem_available_kib_at_start": _meminfo_kib("MemAvailable"),
        "load_average_at_start": os.getloadavg(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": probe[0],
        "gil_enabled": probe[1] == 1,
        # Build string (compiler, build date) and resolved executable: two
        # builds of one version are different interpreters.
        "python_build": probe[2],
        "python_executable": probe[3],
        "work_filesystem": _filesystem(config.work),
        # Same filesystem type on another device (NVMe vs loop or network
        # ext4) is not the same storage.
        "work_device": _backing_device(config.work),
    }


def candidate_identity(candidate: Path) -> dict[str, Any]:
    head = _git(candidate, "rev-parse", "HEAD")
    diff = _git(candidate, "diff", "HEAD", "--binary")
    digest = hashlib.sha256(diff.encode())
    # Untracked, unignored files can be imported too (a new module a tracked
    # edit refers to), so their names and bytes are part of the identity.
    untracked = sorted(_git(candidate, "ls-files", "--others", "--exclude-standard", "-z").split("\0"))
    for name in (name for name in untracked if name):
        digest.update(b"\0untracked\0" + name.encode())
        try:
            # Streamed: an untracked artifact may be larger than memory.
            with (candidate / name).open("rb") as stream:
                while chunk := stream.read(1 << 20):
                    digest.update(chunk)
        except OSError:
            digest.update(b"\0unreadable")
    dirty = bool(diff) or any(untracked)
    return {
        "git_sha": head,
        "dirty": dirty,
        # Edits are part of what runs: a dirty tree edited again during the
        # build must not compare equal to itself.
        "tracked_diff_sha256": digest.hexdigest() if dirty else None,
    }


def candidate_stamp(candidate: Path) -> dict[str, tuple[int, ...]]:
    """Stamps of every tracked and untracked, unignored file and their directories.

    A file stamps as ``(inode, ctime_ns)``: an edit changes its ctime even
    when its bytes are restored before the run ends. Each directory holding
    one, and the root, stamps as ``(inode, mtime_ns, ctime_ns)``: a module
    created, imported and removed again changes its directory's times even
    though neither ``git ls-files`` snapshot lists it. Equal stamps mean the
    daemon could import nothing the recorded identity does not describe. A
    deleted tracked file stamps as ``(0, 0)``.
    """
    names = sorted(
        {
            name
            for name in _git(candidate, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0")
            if name
        }
    )
    stamps: dict[str, tuple[int, ...]] = {}
    directories: set[str] = {"."}
    for name in names:
        directories.update(parent.as_posix() for parent in Path(name).parents)
        try:
            status = os.lstat(candidate / name)
        except FileNotFoundError:
            stamps[name] = (0, 0)
            continue
        stamps[name] = (status.st_ino, status.st_ctime_ns)
    for directory in sorted(directories):
        try:
            status = os.lstat(candidate / directory)
        except FileNotFoundError:
            stamps[f"{directory}/"] = (0, 0, 0)
            continue
        stamps[f"{directory}/"] = (status.st_ino, status.st_mtime_ns, status.st_ctime_ns)
    return stamps


# ---------------------------------------------------------------------------
# run


_SAMPLER_PATH: Final = Path(__file__).with_name("sampler.py")


def _daemon_command(config: RunConfig) -> list[str]:
    # The sampler is loaded by file path from the benchmark's own tree, so a
    # candidate that predates this benchmark (master, an older head) runs with
    # the same instrumentation as the branch under test.
    bootstrap = (
        "import os, runpy\n"
        "sampler = os.environ.get('POLYLOGUE_BENCH_SAMPLER')\n"
        "if sampler: runpy.run_path(sampler)['start_from_environment']()\n"
        "from polylogue.daemon.commands import main\n"
        "main()"
    )
    return [
        config.python,
        "-c",
        bootstrap,
        "run",
        "--no-browser-capture",
        "--no-api",
        "--cold-build-index",
    ]


def _daemon_env(config: RunConfig, paths: dict[str, Path]) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "HOME": str(paths["home"]),
        "XDG_CONFIG_HOME": str(paths["xdg"] / "config"),
        "XDG_DATA_HOME": str(paths["xdg"] / "data"),
        "XDG_STATE_HOME": str(paths["xdg"] / "state"),
        "XDG_CACHE_HOME": str(paths["xdg"] / "cache"),
        "XDG_RUNTIME_DIR": str(paths["xdg"] / "runtime"),
        "TMPDIR": str(paths["tmp"]),
        "POLYLOGUE_ARCHIVE_ROOT": str(paths["archive"]),
        "POLYLOGUE_CONFIG": str(paths["config"]),
        # An empty site layer: a host ``/etc/polylogue/polylogue.toml`` would
        # otherwise add live sources or resource policy outside the receipt.
        "POLYLOGUE_SITE_CONFIG": "",
        "POLYLOGUE_SINEX_MODE": "off",
        "POLYLOGUE_LOG_FORMAT": "json",
        "POLYLOGUE_LOG_FILE": str(paths["events"]),
        "PYTHONPATH": str(config.candidate),
        # Bytecode goes to scratch, never into the candidate tree: the tree's
        # directory stamps (``candidate_stamp``) then move only when files
        # are added or removed there.
        "PYTHONPYCACHEPREFIX": str(paths["tmp"] / "pycache"),
    }
    # Per-thread CPU is always sampled (cheap); stacks only with --profile.
    env["POLYLOGUE_BENCH_SAMPLER"] = str(_SAMPLER_PATH)
    env["POLYLOGUE_BENCH_STACK_SAMPLES"] = str(paths["stacks"])
    if config.profile:
        env["POLYLOGUE_BENCH_STACK_INTERVAL_S"] = str(config.profile_interval_s)
    else:
        env["POLYLOGUE_BENCH_STACKS"] = "0"
        env["POLYLOGUE_BENCH_STACK_INTERVAL_S"] = "0.05"
    # An override may tune the daemon, never redirect what the driver
    # isolates or measures (archive root, config, logs, sampler).
    if owned := sorted(key for key, _value in config.extra_env if key in env):
        raise ValueError(f"--env may not override driver-owned variables: {', '.join(owned)}")
    env.update(dict(config.extra_env))
    return env


def _prepare_paths(config: RunConfig) -> dict[str, Path]:
    work = config.work.absolute()
    if work.exists() and any(work.iterdir()):
        raise ValueError(f"work directory must be absent or empty: {work}")
    home = (config.corpus / "home").resolve(strict=True)
    paths = {
        "work": work,
        "home": home,
        "archive": work / "archive",
        "xdg": work / "xdg",
        "tmp": work / "tmp",
        "events": work / "events.jsonl",
        "stacks": work / "stacks.json",
        "daemon_log": work / "daemon.log",
        "receipt": work / "receipt.json",
    }
    # The archive holds copies of private transcripts: the whole work tree is
    # owner-only before the daemon creates anything in it.
    work.mkdir(parents=True, exist_ok=True)
    work.chmod(0o700)
    for name in ("archive", "tmp"):
        paths[name].mkdir(parents=True, mode=0o700)
    for name in ("config", "data", "state", "cache", "runtime"):
        (paths["xdg"] / name).mkdir(parents=True)
    (paths["xdg"] / "runtime").chmod(0o700)
    # Export archives (ChatGPT, Claude.ai) have no watched location of their
    # own: the operator stages them into the archive inbox with ``polylogue
    # import``, the one route that admits them. The run stages each sealed
    # export file there before the daemon starts, as a copy (a hard link would
    # move the sealed file's ctime and read as a corpus write).
    exports = config.corpus / "exports"
    if exports.is_dir():
        inbox = paths["archive"] / "inbox"
        for path in sorted(exports.iterdir()):
            # Only sealed, real export directories are sources: a symlink
            # (or unknown child) would point the daemon at unsealed files the
            # manifest never hashed.
            if path.is_symlink() or not path.is_dir() or path.name not in EXPORT_ORIGINS:
                raise ValueError(f"corpus export root is not a sealed export directory: {path}")
            for file in sorted(path.iterdir()):
                if file.is_symlink() or not file.is_file():
                    raise ValueError(f"corpus export is not a sealed regular file: {file}")
                inbox.mkdir(mode=0o700, exist_ok=True)
                # Staged under its own name, as ``polylogue import`` stages it.
                staged = inbox / file.name
                if staged.exists():
                    raise ValueError(f"two corpus exports share the name {file.name!r}; rename one")
                shutil.copyfile(file, staged)
    # Hook carriers and legacy pending envelopes live under the archive root,
    # so stage their sealed corpus copy into this run's isolated archive before
    # the daemon starts. No live operator spool is ever opened by the daemon.
    hook_spool = paths["home"] / ".polylogue-hook-spool"
    if hook_spool.is_dir():
        shutil.copytree(hook_spool, paths["archive"] / "hooks", copy_function=shutil.copy2)
    config_path = paths["xdg"] / "config" / "polylogue" / "polylogue.toml"
    config_path.parent.mkdir(parents=True)
    # Embeddings are external API work and stay off.
    config_path.write_text("[embedding]\nenabled = false\n", encoding="utf-8")
    paths["config"] = config_path
    return paths


def _archive_write_stamp(archive: Path) -> tuple[tuple[str, int, int], ...]:
    """Size and mtime of owned databases and write sidecars, for shutdown progress.

    ``-shm`` files are left out: readers update their read marks there.
    Inspect only tier anchors and direct generation members: walking blobs or
    capture payloads every shutdown poll would perturb the measured workload.
    """
    from polylogue.storage.archive_identity import GENERATIONS_DIRNAME, TIER_FILENAMES

    databases = [archive / filename for _, filename in TIER_FILENAMES]
    # Both lifecycle owners place their database immediately below each
    # generation directory, including inactive and retiring generations.
    for dirname, filename in (
        (GENERATIONS_DIRNAME, "index.db"),
        (".embeddings-generations", "embeddings.db"),
    ):
        try:
            members = tuple((archive / dirname).iterdir())
        except OSError:
            continue
        databases.extend(member / filename for member in members)

    stamp: list[tuple[str, int, int]] = []
    for path in sorted(Path(str(database) + suffix) for database in databases for suffix in ("", "-wal", "-journal")):
        try:
            stat = path.stat()
        except OSError:
            continue
        stamp.append((str(path.relative_to(archive)), stat.st_size, stat.st_mtime_ns))
    return tuple(stamp)


def _stop(
    process: subprocess.Popen[bytes],
    *,
    stall_s: float,
    progress: Callable[[], object],
    interrupted: list[int],
    poll_s: float = 5.0,
) -> tuple[int | None, float]:
    """Ask the daemon to stop and wait for it while its shutdown progresses.

    Draining a large archive or checkpointing SQLite can take longer than any
    fixed deadline; killing it then turns a valid build into an unclean
    shutdown. The daemon is killed only when its process tree stops moving
    (``progress`` -- CPU and I/O -- unchanged for ``stall_s``) or when a
    cancellation arrives during the shutdown itself.
    """
    began = time.monotonic()
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        cancellations = len(interrupted)
        last = progress()
        last_moved = time.monotonic()
        while True:
            try:
                process.wait(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            current = progress()
            if current != last:
                last, last_moved = current, now
            if len(interrupted) > cancellations or now - last_moved > stall_s:
                process.kill()
                process.wait()
                break
    return process.returncode, time.monotonic() - began


def run_build(config: RunConfig, *, progress: Callable[[str], None] = print) -> dict[str, Any]:
    manifest = load_manifest(config.corpus)
    paths = _prepare_paths(config)
    hook_preparation = _prepare_hook_spool(paths, progress=progress)
    # Stamped before the identity is read, so an edit between the two is
    # still a changed stamp at the end.
    candidate_files = candidate_stamp(config.candidate)
    identity = candidate_identity(config.candidate)
    env_summary = environment(config)
    command = _daemon_command(config)
    daemon_env = _daemon_env(config, paths)
    interrupted: list[int] = []

    def _interrupt(signum: int, _frame: object) -> None:
        # A cancelled job still owes a receipt for the work it measured. The
        # handler stays installed until the receipt is on disk.
        interrupted.append(signum)

    previous_handlers = {sig: signal.signal(sig, _interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        return _measure_and_write_receipt(
            config,
            manifest=manifest,
            paths=paths,
            identity=identity,
            candidate_files=candidate_files,
            env_summary=env_summary,
            command=command,
            daemon_env=daemon_env,
            interrupted=interrupted,
            progress=progress,
            hook_preparation=hook_preparation,
        )
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def _measure_and_write_receipt(
    config: RunConfig,
    *,
    manifest: dict[str, Any],
    paths: dict[str, Path],
    identity: dict[str, Any],
    candidate_files: dict[str, tuple[int, ...]],
    env_summary: dict[str, Any],
    command: list[str],
    daemon_env: dict[str, str],
    interrupted: list[int],
    progress: Callable[[str], None],
    hook_preparation: dict[str, Any] | None,
) -> dict[str, Any]:
    from devtools.fresh_build_bench.report import build_receipt

    log_stream = paths["daemon_log"].open("wb")
    corpus_stamp = change_stamp(config.corpus)
    started_wall = time.time()
    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=config.candidate, env=daemon_env, stdout=log_stream, stderr=subprocess.STDOUT
    )
    sampler = TreeSampler(process.pid, origin=started)
    sampler.start()
    observations: list[Observation] = []
    # Every exit from the observation loop names its own outcome; a build
    # that keeps making observable progress runs until it settles, and one
    # that stops moving is ``stalled`` by the progress check.
    outcome = "interrupted"
    terminal_at: float | None = None
    promoted_at: float | None = None
    stable = 0
    last_report = 0.0
    last_observation: Observation | None = None
    last_progress_at = 0.0
    retry_opportunity_at: float | None = None
    last_useful_progress_at: float | None = None
    last_activity_at: float | None = None
    work_progress = WorkProgressTail(paths["events"])
    final_work_progress = 0
    # Wall minus monotonic elapsed, sampled every poll: a step that is
    # restored before the end still displaced the milestones logged meanwhile.
    clock_steps: list[float] = [0.0]
    try:
        while True:
            if interrupted:
                outcome = "interrupted"
                break
            if process.poll() is not None:
                outcome = "daemon_exited"
                break
            # A cached readiness census may be no older than half the stall
            # window: a domain turning ready right after one census must be
            # seen before the no-progress deadline can fire.
            observation = observe(
                paths["archive"], started, readiness_max_age_s=min(_READINESS_POLL_S, config.stall_timeout_s / 2)
            )
            observation.work_progress = work_progress.poll()
            if work_progress.convergence_next_run_at is not None:
                observation.convergence_next_run_at = work_progress.convergence_next_run_at
            observations.append(observation)
            clock_steps.append((time.time() - started_wall) - (time.monotonic() - started))
            if observation.error is not None and not observation.error_retryable:
                # The archive cannot be read by this driver at all (a
                # candidate whose schema predates a column it reads): a typed
                # refusal, not a wait that only cancellation could end.
                outcome = "observation_refused"
                break
            if observation.error is not None:
                # A failed read (a busy database) says nothing about progress:
                # its all-zero counts must not alternate with the real ones.
                time.sleep(config.poll_s)
                continue
            useful_progress = _useful_progress(last_observation, observation)
            if last_observation is None or useful_progress:
                last_progress_at = observation.t
                retry_opportunity_at = None
            if useful_progress:
                last_useful_progress_at = observation.t
            if (
                useful_progress
                or last_observation is not None
                and (
                    observation.debt_attempts != last_observation.debt_attempts
                    or observation.cursor_failures != last_observation.cursor_failures
                )
            ):
                last_activity_at = observation.t
            observation.useful_progress_at_s = last_useful_progress_at
            observation.activity_at_s = last_activity_at
            last_observation = observation
            # Preserve declared recovery backoff without recording waiting or
            # failed attempts as new useful progress.
            waiting = bool(observation.debt_waiting_by_stage or observation.cursor_retry_waiting)
            if (
                retry_opportunity_at is None
                and observation.debt_by_stage.get("raw_frontier_inspection", 0)
                and observation.frontier_retry_at is not None
                and observation.convergence_next_run_at is not None
            ):
                due = datetime.fromisoformat(observation.frontier_retry_at).timestamp()
                if observation.convergence_next_run_at >= due:
                    # One actual owner cadence opportunity per useful-progress
                    # epoch. Failed attempts and later schedules cannot renew it.
                    retry_opportunity_at = observation.t + observation.convergence_next_run_at - time.time()
            stall_origin = max(last_progress_at, retry_opportunity_at or last_progress_at)
            if not useful_progress and not waiting and observation.t - stall_origin > config.stall_timeout_s:
                # Nothing observable moved: a starved backlog or a refused
                # promotion. Stop and report it rather than burning the
                # whole timeout on a build that is not converging.
                outcome = "stalled"
                break
            if observation.promoted_index is not None and promoted_at is None:
                promoted_at = observation.t
            if observation.terminal:
                stable += 1
                if stable == 1:
                    terminal_at = observation.t
                if stable >= config.stable_polls:
                    outcome = "terminal"
                    break
            else:
                stable = 0
                terminal_at = None
            if observation.t - last_report >= 30:
                last_report = observation.t
                progress(
                    f"[{observation.t:8.1f}s] cursors {observation.cursor_complete}/{observation.cursor_rows}"
                    f" raw {observation.raw_rows} pending {observation.raw_pending}"
                    f" debt {observation.open_debt} promoted={observation.promoted_index is not None}"
                )
            time.sleep(config.poll_s)
    finally:
        try:
            exit_code, shutdown_s = _stop(
                process,
                stall_s=config.stall_timeout_s,
                # A draining or checkpointing shutdown moves the archive's
                # database and WAL files. Process CPU, read I/O, thread counts and
                # the event log are left out: the stack sampler, status readers
                # and periodic skip events keep those moving in a hung daemon, so
                # a stalled run was never terminated.
                progress=lambda: _archive_write_stamp(paths["archive"]),
                interrupted=interrupted,
            )
        finally:
            try:
                sampler.finish()
            finally:
                try:
                    log_stream.close()
                finally:
                    try:
                        final_work_progress = work_progress.poll()
                    finally:
                        work_progress.close()
    if interrupted:
        # A cancellation after the loop settled (during shutdown, or before
        # the fingerprint) still skips post-processing; the receipt says so.
        outcome = "interrupted"
    finished = time.monotonic()
    finished_wall = time.time()
    final = observe(paths["archive"], started)
    final.work_progress = final_work_progress
    final.useful_progress_at_s = last_useful_progress_at
    final.activity_at_s = last_activity_at
    # The watcher may have read a file edited after the launch-time check.
    try:
        verify_manifest(config.corpus, manifest)
        # Content matching at both ends is not enough: a file written during
        # the run and restored before its end changed ctime.
        corpus_unchanged = change_stamp(config.corpus) == corpus_stamp
    except (OSError, ValueError):
        corpus_unchanged = False
    # Lazy imports run whatever the candidate tree holds when they execute; a
    # checkout or commit during the build makes the recorded SHA a guess.
    # Endpoint equality misses an edit restored before the end, which the
    # daemon may already have imported; file ctimes do not.
    identity["unchanged_during_run"] = (
        candidate_identity(config.candidate)
        == {key: identity[key] for key in ("git_sha", "dirty", "tracked_diff_sha256")}
        and candidate_stamp(config.candidate) == candidate_files
    )
    # A cancellation that arrived after the loop skips the fingerprint, the
    # one step that scales with the archive, as an in-loop one does.
    receipt = build_receipt(
        config=replace(config, fingerprint=False) if interrupted else config,
        manifest=manifest,
        paths=paths,
        identity=identity,
        environment=env_summary,
        command=command,
        started_wall=started_wall,
        wall_s=finished - started,
        outcome=outcome,
        terminal_at=terminal_at,
        exit_code=exit_code,
        shutdown_s=shutdown_s,
        observations=observations,
        final=final,
        tree_samples=sampler.samples,
        daemon_rss_hwm_bytes=sampler.daemon_rss_hwm_bytes,
        corpus_unchanged=corpus_unchanged,
        clock_step_s=max([*clock_steps, (finished_wall - started_wall) - (finished - started)], key=abs),
        cancelled=lambda: bool(interrupted),
    )
    if hook_preparation is not None:
        receipt["hook_preparation"] = hook_preparation
        receipt["hook_end_to_end_s"] = float(receipt["timing_s"]["wall"]) + hook_preparation["compaction_s"]
    # Atomic: a receipt is either absent or complete.
    staging = paths["receipt"].with_name(paths["receipt"].name + ".tmp")
    staging.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, paths["receipt"])
    return receipt


def _prepare_hook_spool(paths: dict[str, Path], *, progress: Callable[[str], None] = print) -> dict[str, Any] | None:
    """Run the production legacy-spool compactor in the isolated archive."""
    root = paths["archive"] / "hooks"
    pending = root / "pending"
    if not pending.is_dir():
        return None
    from polylogue.sources.hook_producer import compact_legacy_spool

    started = time.monotonic()
    result = compact_legacy_spool(root)
    compaction_s = time.monotonic() - started
    summary = {"compaction_s": compaction_s, "result": result}
    refused = result.get("refused", {})
    refused_count = (
        sum(value for value in refused.values() if isinstance(value, int)) if isinstance(refused, dict) else 0
    )
    progress(
        "hook spool compaction "
        f"seconds={compaction_s:.3f} "
        f"scanned={result.get('scanned', 0)} "
        f"folded={result.get('folded', 0)} "
        f"refused={refused_count}"
    )
    return summary


__all__ = ["Observation", "RunConfig", "TreeSampler", "observe", "run_build"]
