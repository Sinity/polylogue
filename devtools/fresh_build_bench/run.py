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
import os
import platform
import signal
import sqlite3
import subprocess
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
    #: Sum of retry attempts over open debt: a scheduled retry that fails
    #: again still moves it.
    debt_attempts: int = 0
    #: Summed cursor failures, and cursors waiting on a scheduled retry.
    cursor_failures: int = 0
    cursor_retry_waiting: int = 0
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
            if "convergence_debt" in tables:
                for stage, count in conn.execute("SELECT stage, COUNT(*) FROM convergence_debt GROUP BY stage"):
                    observation.debt_by_stage[str(stage)] = int(count)
                observation.open_debt = sum(observation.debt_by_stage.values())
                if "ingest_cursor" in tables:
                    cursor_row = conn.execute(
                        "SELECT COALESCE(SUM(failure_count), 0),"
                        " COALESCE(SUM(next_retry_at IS NOT NULL AND next_retry_at > ?), 0) FROM ingest_cursor",
                        (datetime.now(UTC).isoformat(),),
                    ).fetchone()
                    observation.cursor_failures, observation.cursor_retry_waiting = (int(v) for v in cursor_row)
                observation.debt_attempts = int(
                    conn.execute("SELECT COALESCE(SUM(attempts), 0) FROM convergence_debt").fetchone()[0]
                )
                now_iso = datetime.now(UTC).isoformat()
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


def _retryable_observation_error(exc: BaseException) -> bool:
    if isinstance(exc, sqlite3.OperationalError):
        message = str(exc).lower()
        return any(token in message for token in ("locked", "busy", "unable to open", "disk i/o"))
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
        "from polylogue.daemon.cli import main\n"
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
    # Export archives (ChatGPT, Claude.ai) are not typed default sources; the
    # operator declares their directories as additional roots, and so does
    # this config. Embeddings are external API work and stay off.
    exports = config.corpus / "exports"
    roots: list[str] = []
    if exports.is_dir():
        for path in sorted(exports.iterdir()):
            # Only sealed, real export directories are sources: a symlink
            # (or unknown child) would point the daemon at unsealed files the
            # manifest never hashed.
            if path.is_symlink() or not path.is_dir() or path.name not in EXPORT_ORIGINS:
                raise ValueError(f"corpus export root is not a sealed export directory: {path}")
            roots.append(str(path))
    config_path = paths["xdg"] / "config" / "polylogue" / "polylogue.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "[sources]\nroots = " + json.dumps(roots) + "\n\n[embedding]\nenabled = false\n", encoding="utf-8"
    )
    paths["config"] = config_path
    return paths


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
    last_progress_key: tuple[object, ...] | None = None
    last_progress_at = 0.0
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
            observations.append(observation)
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
            progress_key = (
                observation.cursor_rows,
                observation.cursor_complete,
                observation.raw_rows,
                observation.raw_pending,
                observation.memberships_pending,
                observation.open_debt,
                # A stage that resolves one debt row while the next stage
                # creates another leaves the total unchanged even though the
                # daemon is actively converging; the per-stage breakdown
                # moves and must count as progress too.
                tuple(sorted(observation.debt_by_stage.items())),
                # Each scheduled retry attempt, even one that fails again.
                observation.debt_attempts,
                observation.cursor_failures,
                observation.promoted_index,
                # Derived convergence after promotion may move nothing but
                # readiness; each domain turning ready is progress.
                tuple(sorted(observation.readiness.items())),
            )
            if progress_key != last_progress_key:
                last_progress_key, last_progress_at = progress_key, observation.t
            elif observation.debt_waiting_by_stage or observation.cursor_retry_waiting:
                # Debt waits on its scheduled retry (the production backoff
                # reaches 960 s, beyond the stall window): waiting on the
                # schedule is not a stall.
                last_progress_at = observation.t
            elif observation.t - last_progress_at > config.stall_timeout_s:
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
        exit_code, shutdown_s = _stop(
            process,
            stall_s=config.stall_timeout_s,
            progress=lambda: sampler.samples[-1][2:] if sampler.samples else None,
            interrupted=interrupted,
        )
        sampler.finish()
        log_stream.close()
    if interrupted:
        # A cancellation after the loop settled (during shutdown, or before
        # the fingerprint) still skips post-processing; the receipt says so.
        outcome = "interrupted"
    finished = time.monotonic()
    finished_wall = time.time()
    final = observe(paths["archive"], started)
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
        clock_step_s=(finished_wall - started_wall) - (finished - started),
        cancelled=lambda: bool(interrupted),
    )
    # Atomic: a receipt is either absent or complete.
    staging = paths["receipt"].with_name(paths["receipt"].name + ".tmp")
    staging.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, paths["receipt"])
    return receipt


__all__ = ["Observation", "RunConfig", "TreeSampler", "observe", "run_build"]
