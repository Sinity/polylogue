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

from devtools.fresh_build_bench.corpus import load_manifest, verify_manifest

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
    timeout_s: float = 6 * 3600.0
    settle_timeout_s: float = 1800.0
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
    promoted_index: str | None = None
    readiness: dict[str, bool] = field(default_factory=dict)
    error: str | None = None

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


def observe(archive: Path, started: float) -> Observation:
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
            from polylogue.storage.archive_readiness import archive_readiness_status

            readiness = archive_readiness_status(archive)
            surfaces = readiness.get("surfaces", {}) if readiness.get("checked") is True else {}
            observation.readiness = {
                str(name): isinstance(surface, dict) and surface.get("ready") is True
                for name, surface in surfaces.items()
            }
    except (OSError, sqlite3.Error) as exc:
        observation.error = f"{type(exc).__name__}: {exc}"
    return observation


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


def _meminfo_kib(key: str) -> int | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith(key + ":"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


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
    return {
        "host_cpu_count": os.cpu_count(),
        "host_mem_total_kib": _meminfo_kib("MemTotal"),
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
            digest.update((candidate / name).read_bytes())
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
        "POLYLOGUE_SINEX_MODE": "off",
        "POLYLOGUE_LOG_FORMAT": "json",
        "POLYLOGUE_LOG_FILE": str(paths["events"]),
        "PYTHONPATH": str(config.candidate),
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
    roots = sorted(str(path.resolve()) for path in exports.iterdir() if path.is_dir()) if exports.is_dir() else []
    config_path = paths["xdg"] / "config" / "polylogue" / "polylogue.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "[sources]\nroots = " + json.dumps(roots) + "\n\n[embedding]\nenabled = false\n", encoding="utf-8"
    )
    paths["config"] = config_path
    return paths


def _stop(process: subprocess.Popen[bytes], timeout_s: float) -> tuple[int | None, float]:
    began = time.monotonic()
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    return process.returncode, time.monotonic() - began


def run_build(config: RunConfig, *, progress: Callable[[str], None] = print) -> dict[str, Any]:
    manifest = load_manifest(config.corpus)
    paths = _prepare_paths(config)
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
    env_summary: dict[str, Any],
    command: list[str],
    daemon_env: dict[str, str],
    interrupted: list[int],
    progress: Callable[[str], None],
) -> dict[str, Any]:
    from devtools.fresh_build_bench.report import build_receipt

    log_stream = paths["daemon_log"].open("wb")
    started_wall = time.time()
    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=config.candidate, env=daemon_env, stdout=log_stream, stderr=subprocess.STDOUT
    )
    sampler = TreeSampler(process.pid, origin=started)
    sampler.start()
    observations: list[Observation] = []
    outcome = "timeout"
    terminal_at: float | None = None
    promoted_at: float | None = None
    stable = 0
    last_report = 0.0
    last_progress_key: tuple[object, ...] | None = None
    last_progress_at = 0.0
    try:
        deadline = started + config.timeout_s
        while time.monotonic() < deadline:
            if interrupted:
                outcome = "interrupted"
                break
            if process.poll() is not None:
                outcome = "daemon_exited"
                break
            observation = observe(paths["archive"], started)
            observations.append(observation)
            progress_key = (
                observation.cursor_rows,
                observation.cursor_complete,
                observation.raw_rows,
                observation.raw_pending,
                observation.memberships_pending,
                observation.open_debt,
                observation.promoted_index,
                # Derived convergence after promotion may move nothing but
                # readiness; each domain turning ready is progress.
                tuple(sorted(observation.readiness.items())),
            )
            if progress_key != last_progress_key:
                last_progress_key, last_progress_at = progress_key, observation.t
            elif observation.t - last_progress_at > config.stall_timeout_s:
                # Nothing observable moved: a starved backlog or a refused
                # promotion. Stop and report it rather than burning the
                # whole timeout on a build that is not converging.
                outcome = "stalled"
                break
            if observation.promoted_index is not None and promoted_at is None:
                promoted_at = observation.t
            if promoted_at is not None and observation.t - promoted_at > config.settle_timeout_s:
                outcome = "settle_timeout"
                break
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
        exit_code, shutdown_s = _stop(process, 300.0)
        sampler.finish()
        log_stream.close()
    finished = time.monotonic()
    final = observe(paths["archive"], started)
    # The watcher may have read a file edited after the launch-time check.
    try:
        verify_manifest(config.corpus, manifest)
        corpus_unchanged = True
    except (OSError, ValueError):
        corpus_unchanged = False
    # Lazy imports run whatever the candidate tree holds when they execute; a
    # checkout or commit during the build makes the recorded SHA a guess.
    identity["unchanged_during_run"] = candidate_identity(config.candidate) == {
        key: identity[key] for key in ("git_sha", "dirty", "tracked_diff_sha256")
    }
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
    )
    # Atomic: a receipt is either absent or complete.
    staging = paths["receipt"].with_name(paths["receipt"].name + ".tmp")
    staging.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, paths["receipt"])
    return receipt


__all__ = ["Observation", "RunConfig", "TreeSampler", "observe", "run_build"]
