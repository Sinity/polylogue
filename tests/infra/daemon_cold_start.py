"""Private, bounded observation of a real cold ``polylogued run`` process."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from devtools.isolated_environment import isolated_home_environment
from devtools.measurement_receipts import emit_receipt

SESSION_IDS = tuple(f"claude-code-session:ccccd500-0000-0000-0000-{i:012d}" for i in range(3))
SEARCH_TOKEN = "coldqualificationuniquetoken"
DEADLINE_S = 120.0
RSS_TREE_MAX_PROCESSES = 1024
RSS_TREE_MAX_TASK_IDS = 4096
RSS_PROC_READ_LIMIT_BYTES = 4096


def _proc_children_supported(proc_root: Path = Path("/proc")) -> bool:
    """Whether this procfs exposes per-task child lists for the current task."""
    probe = proc_root / str(os.getpid()) / "task" / str(threading.get_native_id()) / "children"
    return probe.is_file()


#: The fixture is presented as Claude Code's projects directory, so it uses
#: that declared layout: sessions sit directly in a project directory (named,
#: as Claude Code names them, after a working directory with ``-`` for ``/``).
#: The rejected prefix is root-level files whose names sort before every
#: project (``+`` sorts below ``-``), and the third session lives in a second
#: project that sorts after the first.
FIXTURE_PROJECT = "-a-cold-qualification"
FIXTURE_NESTED_PROJECT = "-z-cold-qualification"
FIXTURE_NESTED_SESSION = f"{FIXTURE_NESTED_PROJECT}/z-session-2.jsonl"


def write_fixture(root: Path, *, rejected: int, malformed_last: bool = False) -> str:
    root.mkdir(parents=True)
    project = root / FIXTURE_PROJECT
    project.mkdir()
    for i in range(rejected):
        path = root / f"+rejected-{i:05d}.txt"
        data = b"synthetic rejected entry\n"
        path.write_bytes(data)
    for i, session_id in enumerate(SESSION_IDS):
        path = (root / FIXTURE_NESTED_PROJECT if i == 2 else project) / f"z-session-{i}.jsonl"
        path.parent.mkdir(exist_ok=True)
        if malformed_last and i == 2:
            data = b'{"sessionId": "invalid", "message": \n'
        else:
            native = session_id.split(":", 1)[1]
            lines = []
            for n in range(12):
                role = "user" if n % 2 == 0 else "assistant"
                uuid = f"msg-{native}-{n:03d}"
                lines.append(
                    json.dumps(
                        {
                            "parentUuid": None if n == 0 else f"msg-{native}-{n - 1:03d}",
                            "sessionId": native,
                            "type": role,
                            "message": {"role": role, "content": f"{SEARCH_TOKEN} synthetic message {n}"},
                            "uuid": uuid,
                            "timestamp": f"2026-05-20T00:00:{n:02d}.000Z",
                            "cwd": "/synthetic/cold-qualification",
                            "version": "1.0.6",
                            "isSidechain": False,
                            "userType": "external",
                        },
                        sort_keys=True,
                    )
                )
            data = ("\n".join(lines) + "\n").encode()
        path.write_bytes(data)
    return fixture_tree_digest(root)


def fixture_tree_digest(root: Path) -> str:
    """Hash fixture paths and file bytes in canonical order for immutability."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix().encode()
        if path.is_symlink():
            digest.update(b"L\0" + relative + b"\0" + os.readlink(path).encode())
        elif path.is_dir():
            digest.update(b"D\0" + relative + b"\0")
        elif path.is_file():
            digest.update(b"F\0" + relative + b"\0" + path.read_bytes())
    return digest.hexdigest()


def write_retained_measurement_receipt(receipt: dict[str, object], report_file: Path | None) -> Path | None:
    """Export sanitized cold-route scalars beside the managed pytest report."""
    if report_file is None:
        return None
    fixture = receipt.get("fixture")
    if not isinstance(fixture, dict) or fixture.get("rejected") != 4096:
        return None
    discovery = receipt.get("discovery_measurement")
    intake = receipt.get("intake_counts")
    candidate = receipt.get("candidate")
    if not isinstance(discovery, dict) or not isinstance(intake, dict) or not isinstance(candidate, dict):
        raise AssertionError("4096-sibling receipt is missing retained measurement evidence")
    survivors = receipt.get("process_tree_survivors")
    survivor_count = len(survivors) if isinstance(survivors, list) else None

    retained = {
        "format": "polylogue.daemon-cold-discovery-retained.v1",
        "candidate_sha": candidate.get("sha"),
        "schema_validation_mode": receipt.get("schema_validation_mode"),
        "workload": {
            "rejected_siblings": fixture.get("rejected"),
            "accepted_sessions": fixture.get("accepted"),
            "source_unchanged": fixture.get("unchanged_after_run"),
            "source_tree_sha256": fixture.get("sha256"),
            "source_tree_sha256_after": fixture.get("sha256_after_run"),
        },
        "outcome": receipt.get("outcome"),
        "outer_elapsed_s": receipt.get("outer_elapsed_s"),
        "discovery": discovery,
        "intake_counts": {
            key: intake.get(key)
            for key in (
                "event",
                "outcome",
                "files",
                "offered_bytes",
                "succeeded",
                "failed",
                "retried",
                "deferred",
                "refused",
            )
        },
        "process_tree_rss": {
            "available": receipt.get("process_tree_rss_available"),
            "sampled_peak_bytes": receipt.get("process_tree_rss_bytes"),
            "sample_count": receipt.get("process_tree_rss_sample_count"),
            "process_count_at_peak": receipt.get("process_tree_rss_process_count_at_peak"),
            "task_count_at_peak": receipt.get("process_tree_rss_task_count_at_peak"),
            "truncated_sample_count": receipt.get("process_tree_rss_truncated_sample_count"),
            "peak_sample_truncated": receipt.get("process_tree_rss_peak_sample_truncated"),
            "sampling_interval_target_ms": receipt.get("process_tree_rss_sampling_interval_target_ms"),
            "missing_reason": receipt.get("process_tree_rss_missing_reason"),
            "scope": receipt.get("process_tree_rss_scope"),
            "limits": receipt.get("process_tree_rss_limits"),
        },
        "process_tree_survivor_check": receipt.get("process_tree_survivor_check"),
        "process_tree_survivor_count": survivor_count,
        "process_tree_survivor_check_missing_reason": receipt.get("process_tree_survivor_check_missing_reason"),
    }
    path = report_file.with_name("cold-daemon-4096-measurement.json")
    path.write_text(json.dumps(retained, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request(base: str, path: str, token: str) -> tuple[int | None, Any, float]:
    started = time.monotonic()
    req = Request(base + path, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    try:
        with urlopen(req, timeout=2.0) as response:
            raw = response.read(2_000_000)
            content_type = response.headers.get("Content-Type", "")
            body = json.loads(raw) if "json" in content_type else raw.decode(errors="replace")
            return response.status, body, time.monotonic() - started
    except HTTPError as exc:
        return exc.code, None, time.monotonic() - started
    except (URLError, TimeoutError, OSError):
        return None, None, time.monotonic() - started


def _tail(path: Path, *, lines: int = 16) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", errors="replace") as stream:
        return list(deque((line.rstrip("\n") for line in stream), maxlen=lines))


def _bounded_status(status: dict[str, object] | None) -> dict[str, object] | None:
    if status is None:
        return None
    catchup = status.get("catchup")
    snapshot = status.get("status_snapshot")
    return {
        "ok": status.get("ok"),
        "run_id": status.get("run_id"),
        "catchup": {
            key: catchup.get(key)
            for key in (
                "mode",
                "current_phase",
                "current_source",
                "discovery_pending",
                "discovery_age_s",
                "discovery_last_advanced_age_s",
                "discovery_inspected_count",
                "discovery_accepted_count",
                "discovery_rejected_count",
                "planned_file_count",
                "eta_s",
                "failed_file_count",
                "cumulative_failed_file_attempts",
            )
        }
        if isinstance(catchup, dict)
        else None,
        "snapshot": {key: snapshot.get(key) for key in ("state", "age_s", "captured_at")}
        if isinstance(snapshot, dict)
        else None,
    }


def _diagnostics(metrics: str | None) -> dict[str, object]:
    if metrics is None:
        return {"state": "unavailable", "reason": "metrics_request_failed"}
    return {
        "state": "measured",
        "delivery": [line for line in metrics.splitlines() if line.startswith("polylogue_diagnostic_delivery_total{")][
            :16
        ],
        "queue_depth": [line for line in metrics.splitlines() if line.startswith("polylogue_diagnostic_queue_depth ")][
            :1
        ],
    }


def _durable_raw_count(archive: Path) -> int | None:
    db = archive / "source.db"
    if not db.exists():
        return None
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=0.1) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0])
    except sqlite3.Error:
        return None


def _intake_page_evidence(event_log: Path) -> dict[str, object] | None:
    """Return the last structured production intake-page outcome, if present."""
    observed: dict[str, object] | None = None
    try:
        with event_log.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict) or event.get("event") != "daemon.intake.page":
                    continue
                observed = {
                    "event": event.get("event"),
                    "run_id": event.get("run_id"),
                    "component": event.get("component"),
                    "outcome": event.get("outcome"),
                    "files": event.get("files"),
                    "offered_bytes": event.get("bytes"),
                    "succeeded": event.get("succeeded"),
                    "failed": event.get("failed"),
                    "retried": event.get("retried"),
                    "deferred": event.get("deferred"),
                    "refused": event.get("refused"),
                    "observed_at": event.get("ts"),
                }
    except OSError:
        return None
    return observed


def _durable_parse_error(archive: Path, source_path: Path) -> str | None:
    db = archive / "source.db"
    if not db.exists():
        return None
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=0.1) as conn:
            row = conn.execute(
                "SELECT parse_error FROM raw_sessions WHERE source_path = ? ORDER BY acquired_at_ms DESC LIMIT 1",
                (str(source_path),),
            ).fetchone()
            return str(row[0]) if row is not None and row[0] else None
    except sqlite3.Error:
        return None


def _process_tree_rss(
    root_pid: int,
    *,
    proc_root: Path = Path("/proc"),
    max_processes: int = RSS_TREE_MAX_PROCESSES,
    max_task_ids: int = RSS_TREE_MAX_TASK_IDS,
    children_file_limit_bytes: int = RSS_PROC_READ_LIMIT_BYTES,
) -> dict[str, Any] | None:
    """Sum RSS while following children reported by every scanned task ID.

    Procfs is a live view: thread and child sets can change during a sample.
    Process and task limits bound each pass, and ``truncated`` makes a capped
    sample explicitly incomplete instead of presenting it as a full tree.
    """
    pending = [root_pid]
    queued = {root_pid}
    seen: set[int] = set()
    rss_total = 0
    measured = 0
    task_count = 0
    truncated = False
    process_identities: list[dict[str, int]] = []
    while pending:
        if len(seen) >= max_processes:
            truncated = True
            break
        pid = pending.pop()
        queued.discard(pid)
        if pid in seen:
            continue
        seen.add(pid)
        process_path = proc_root / str(pid)
        try:
            stat_text = (process_path / "stat").read_text(encoding="ascii")
            stat_fields = stat_text[stat_text.rfind(")") + 2 :].split()
            if len(stat_fields) > 19:
                process_identities.append({"pid": pid, "start_time_ticks": int(stat_fields[19])})
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
            # The PID can exit between discovery and identity capture. It is
            # still safe to report the RSS sample; the survivor check remains
            # bounded by identities actually captured while the tree lived.
            pass
        status_path = process_path / "status"
        found_rss = False
        try:
            with status_path.open(encoding="ascii") as stream:
                for line in itertools.islice(stream, 128):
                    if line.startswith("VmRSS:"):
                        rss_total += int(line.split()[1]) * 1024
                        measured += 1
                        found_rss = True
                        break
        except (FileNotFoundError, ProcessLookupError):
            # A PID reached through a parent's children list may exit before
            # its status is read. Its descendants may still be alive, so the
            # process-tree sample is incomplete even when the PID is gone.
            if pid != root_pid:
                truncated = True
            continue
        except (PermissionError, ValueError, OSError):
            truncated = True
            continue
        if not found_rss:
            truncated = True

        remaining_tasks = max_task_ids - task_count
        if remaining_tasks <= 0:
            truncated = True
            break
        task_ids: list[int] = []
        task_path = process_path / "task"
        try:
            with os.scandir(task_path) as entries:
                for entry in itertools.islice(entries, remaining_tasks + 1):
                    if not entry.name.isdecimal():
                        continue
                    if len(task_ids) == remaining_tasks:
                        truncated = True
                        break
                    task_ids.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            truncated = True
            continue
        except (PermissionError, OSError):
            truncated = True
            continue

        # Include the process leader's task even if procfs reordered a large
        # task directory and the scan cap was reached before its entry.
        if pid not in task_ids:
            if len(task_ids) < remaining_tasks:
                task_ids.insert(0, pid)
            elif task_ids:
                task_ids[-1] = pid
                truncated = True
        task_count += len(task_ids)

        for task_id in task_ids:
            children_path = task_path / str(task_id) / "children"
            try:
                with children_path.open(encoding="ascii") as stream:
                    child_text = stream.read(children_file_limit_bytes + 1)
            except (FileNotFoundError, ProcessLookupError):
                truncated = True
                continue
            except (PermissionError, OSError):
                truncated = True
                continue
            if len(child_text) > children_file_limit_bytes:
                truncated = True
                child_text = child_text[:children_file_limit_bytes]
                # A bounded read can end in the middle of a decimal PID.
                # Discard that suffix instead of treating its prefix as a PID.
                if child_text and not child_text[-1].isspace():
                    child_text = child_text.rsplit(maxsplit=1)[0] if child_text.split()[:-1] else ""
            for child_text_id in child_text.split()[: max_processes + 1]:
                try:
                    child_pid = int(child_text_id)
                except ValueError:
                    continue
                if child_pid in seen or child_pid in queued:
                    continue
                if len(seen) + len(pending) >= max_processes:
                    truncated = True
                    break
                pending.append(child_pid)
                queued.add(child_pid)
    return (
        {
            "rss_bytes": rss_total,
            "process_count": measured,
            "task_count": task_count,
            "truncated": truncated,
            "process_identities": process_identities,
        }
        if measured
        else None
    )


def _owned_process_identity_state(pid: int, start_time_ticks: int, *, proc_root: Path = Path("/proc")) -> str | None:
    """Return a matching process state, or ``None`` when that identity exited."""
    try:
        stat_text = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    except (PermissionError, OSError):
        return "unreadable"
    stat_fields = stat_text[stat_text.rfind(")") + 2 :].split()
    if len(stat_fields) <= 19:
        return "unreadable"
    try:
        current_start_time = int(stat_fields[19])
    except ValueError:
        return "unreadable"
    if current_start_time != start_time_ticks:
        # PID reuse is a different process and is not owned by this daemon.
        return None
    return stat_fields[0] if stat_fields[0] != "Z" else None


def _read_discovery_trace(path: Path, process_start: float, milestones: dict[str, object]) -> dict[str, object]:
    """Normalize test-seam timestamps against process creation."""
    names: dict[str, float] = {}
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            if (
                isinstance(row, dict)
                and isinstance(row.get("name"), str)
                and isinstance(row.get("monotonic"), (int, float))
            ):
                names.setdefault(row["name"], float(row["monotonic"]))
    except (OSError, json.JSONDecodeError):
        rows = []
    elapsed = {name: round(timestamp - process_start, 6) for name, timestamp in names.items()}

    def interval(start_name: str, end_name: str) -> float | None:
        if start_name not in names or end_name not in names:
            return None
        return round(names[end_name] - names[start_name], 6)

    publication = milestones.get("first_publication")
    publication_s = float(publication) if isinstance(publication, (int, float)) else None
    if publication_s is not None:
        elapsed["first_publication_upper_bound"] = round(publication_s, 6)
    first_yield_elapsed = elapsed.get("first_yielded_entry")
    listing_start_elapsed = elapsed.get("root_listing_start")
    return {
        "clock": "host monotonic; child timestamps normalized to immediately-before-Popen parent timestamp",
        "timestamps_elapsed_s": elapsed,
        "intervals_s": {
            "root_listing_and_entry_inspection": interval("root_listing_start", "root_listing_end"),
            "root_sort_after_listing": interval("root_listing_end", "root_sort_end"),
            "root_walk_to_first_yield": interval("root_listing_start", "first_yielded_entry"),
            "first_yield_to_first_publication_upper_bound": round(publication_s - first_yield_elapsed, 6)
            if publication_s is not None and first_yield_elapsed is not None
            else None,
            "root_listing_start_to_first_publication_upper_bound": round(publication_s - listing_start_elapsed, 6)
            if publication_s is not None and listing_start_elapsed is not None
            else None,
        },
        "first_yield_kind": next(
            (row.get("kind") for row in rows if isinstance(row, dict) and row.get("name") == "first_yielded_entry"),
            None,
        ),
        "listing_scope": "time from entering production _ordered_children for source.root through exhaustion of its os.scandir iterator; includes per-entry type checks and child-list construction",
        "sort_scope": "time after scandir iterator exhaustion until production _ordered_children returns its sorted children",
        "first_publication_scope": "first successful exact-session API read, timestamped at response completion; it is an upper bound on publication time",
        "missing_events": [
            name
            for name in ("root_listing_start", "root_listing_end", "root_sort_end", "first_yielded_entry")
            if name not in names
        ],
    }


def qualify(
    *,
    archive: Path,
    source: Path,
    artifacts: Path,
    rejected: int,
    digest: str,
    malformed_last: bool = False,
    held: bool = False,
    measure_discovery: bool = False,
) -> dict[str, Any]:
    """Run one owned process and emit its receipt in ``finally``."""
    artifacts.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    assert not any(archive.iterdir()), "cold archive must have no files at launch"
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    token = "synthetic-cold-qualification-token"
    stderr = artifacts / "daemon.stderr.log"
    event_log = artifacts / "daemon.events.jsonl"
    samples_path = artifacts / "status-samples.json"
    discovery_trace_path = artifacts / "discovery-trace.jsonl"
    marker = artifacts / "discovery-held"
    release = artifacts / "release-discovery"
    config = artifacts / "absent-private-config.toml"
    candidate_root = Path(__file__).resolve().parents[2]
    # Sources are acquired only from canonical locations, so the sealed
    # fixture tree is presented as Claude Code's projects directory of an
    # isolated home; the operator's real provider directories stay unwatched.
    home = artifacts / "home"
    (home / ".claude").mkdir(parents=True, exist_ok=True)
    projects = home / ".claude" / "projects"
    # A reused artifacts directory may hold a link to an earlier (or since
    # moved) fixture; the daemon must watch exactly this qualification's.
    if projects.is_symlink() and projects.resolve() != source.resolve():
        projects.unlink()
    if not projects.is_symlink():
        projects.symlink_to(source, target_is_directory=True)
    # Every discovery root (HOME, the XDG roots, Polylogue path overrides) is
    # pointed into the isolated home: an inherited XDG config or state root
    # would otherwise hand the daemon the operator's Drive credentials.
    env = isolated_home_environment(os.environ, home=home)
    env.update(
        {
            "POLYLOGUE_ARCHIVE_ROOT": str(archive),
            "POLYLOGUE_CONFIG": str(config),
            "POLYLOGUE_SITE_CONFIG": "",
            "POLYLOGUE_LOG_FORMAT": "json",
            "POLYLOGUE_LOG_FILE": str(event_log),
            # workspace_env disables validation for unrelated pipeline tests.
            # A cold daemon qualification must exercise the ordinary default.
            "POLYLOGUE_SCHEMA_VALIDATION": "advisory",
            "PYTHONPATH": str(candidate_root),
        }
    )
    for key in tuple(env):
        if key.startswith("POLYLOGUE_EMBEDDING"):
            env.pop(key)
    command = [sys.executable]
    if held or measure_discovery:
        env.update(
            {
                "COLD_MARKER": str(marker),
                "COLD_RELEASE": str(release),
                "COLD_DISCOVERY_TRACE": str(discovery_trace_path),
                "COLD_HOLD_DISCOVERY": "1" if held else "0",
            }
        )
        command += ["-c", _INSTRUMENTED_BOOTSTRAP]
    else:
        command += ["-c", "from polylogue.daemon.commands import main; main()"]
    command += [
        "run",
        "--no-browser-capture",
        "--api-port",
        str(port),
        "--api-auth-token",
        token,
    ]
    rss_available = _proc_children_supported()
    receipt: dict[str, object] = {
        "format": "polylogue.daemon-cold-qualification.v1",
        "candidate": {
            "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=candidate_root, text=True).strip(),
            "import_root": str(candidate_root),
            "python": sys.executable,
            "version": platform.python_version(),
        },
        "schema_validation_mode": env["POLYLOGUE_SCHEMA_VALIDATION"],
        "fixture": {
            "rejected": rejected,
            "accepted": 3,
            "messages_per_session": 12,
            "nested_sessions": 1,
            "sha256": digest,
            "malformed_last": malformed_last,
        },
        "archive_root": str(archive),
        "source_root": str(source),
        "artifacts": {
            "stderr": str(stderr),
            "events": str(event_log),
            "samples": str(samples_path),
            "discovery_trace": str(discovery_trace_path) if measure_discovery else None,
        },
        "outcome": "setup_failure",
        "milestones_upper_bound_s": {},
        "request": {},
        "phase_coverage": [],
        "internal_intervals": None,
        "internal_intervals_missing_reason": "no_owner_interval_evidence",
        "process_tree_rss_bytes": None,
        "process_tree_rss_available": rss_available,
        "process_tree_rss_missing_reason": None if rss_available else "proc_children_unavailable",
        "process_tree_rss_scope": "sum of VmRSS for the daemon PID and descendants found through children files for every scanned task ID; live procfs races and any cap are reported as truncated samples",
        "process_tree_rss_sample_count": 0,
        "process_tree_rss_process_count_at_peak": None,
        "process_tree_rss_task_count_at_peak": None,
        "process_tree_rss_truncated_sample_count": 0,
        "process_tree_rss_peak_sample_truncated": None,
        "process_tree_rss_sampling_interval_target_ms": 250,
        "process_tree_survivor_check": "pending" if rss_available else "unavailable",
        "process_tree_survivors": [],
        "process_tree_survivor_check_missing_reason": None if rss_available else "proc_children_unavailable",
        "process_tree_rss_limits": {
            "processes_per_sample": RSS_TREE_MAX_PROCESSES,
            "task_ids_per_sample": RSS_TREE_MAX_TASK_IDS,
            "children_file_bytes_per_task": RSS_PROC_READ_LIMIT_BYTES,
            "status_lines_per_process": 128,
        },
    }
    counts: Counter[str] = Counter()
    max_latency: dict[str, float] = {}
    samples: deque[dict[str, object]] = deque(maxlen=128)
    phases: set[str] = set()
    latest_status: dict[str, object] | None = None
    latest_metrics: str | None = None
    proc: subprocess.Popen[bytes] | None = None
    start = time.monotonic()
    deadline = start + DEADLINE_S
    milestone = receipt["milestones_upper_bound_s"]
    assert isinstance(milestone, dict)
    verified: set[str] = set()
    held_checked = False
    held_marker_seen_at: float | None = None
    held_status: dict[str, object] | None = None
    held_status_latency: float | None = None
    held_metrics_latency: float | None = None
    expected_malformed_refusal = False
    durable_raw_count_max = 0
    peak_tree_rss_bytes = 0
    peak_tree_process_count: int | None = None
    peak_tree_task_count: int | None = None
    peak_tree_truncated = False
    rss_samples = 0
    rss_truncated_samples = 0
    observed_process_identities: dict[int, int] = {}
    rss_stop = threading.Event()
    rss_thread: threading.Thread | None = None

    def retain_tree_identities(tree_sample: dict[str, Any] | None) -> None:
        if tree_sample is None:
            return
        identities = tree_sample.get("process_identities")
        if not isinstance(identities, list):
            return
        for identity in identities:
            if (
                isinstance(identity, dict)
                and isinstance(identity.get("pid"), int)
                and isinstance(identity.get("start_time_ticks"), int)
            ):
                observed_process_identities.setdefault(identity["pid"], identity["start_time_ticks"])

    def sample_owned_tree(root_pid: int) -> None:
        nonlocal peak_tree_rss_bytes, peak_tree_process_count, peak_tree_task_count
        nonlocal peak_tree_truncated, rss_samples, rss_truncated_samples
        while not rss_stop.is_set():
            tree_rss = _process_tree_rss(root_pid)
            if tree_rss is not None:
                retain_tree_identities(tree_rss)
                rss_samples += 1
                if tree_rss["truncated"]:
                    rss_truncated_samples += 1
                if tree_rss["rss_bytes"] > peak_tree_rss_bytes:
                    peak_tree_rss_bytes = tree_rss["rss_bytes"]
                    peak_tree_process_count = tree_rss["process_count"]
                    peak_tree_task_count = tree_rss["task_count"]
                    peak_tree_truncated = bool(tree_rss["truncated"])
            rss_stop.wait(0.25)

    error: str | None = None
    try:
        with stderr.open("wb") as stream:
            # Keep the parent and instrumented child on the same host monotonic
            # clock, with process creation as the outer timing origin.
            start = time.monotonic()
            deadline = start + DEADLINE_S
            proc = subprocess.Popen(command, cwd=candidate_root, env=env, stdout=stream, stderr=subprocess.STDOUT)
            receipt["pid"] = proc.pid
            receipt["outcome"] = "incomplete_population"
            if receipt["process_tree_rss_available"]:
                rss_thread = threading.Thread(target=sample_owned_tree, args=(proc.pid,), daemon=True)
                rss_thread.start()
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    receipt["outcome"] = "daemon_exit"
                    raise AssertionError(f"daemon exited before completion: {proc.returncode}")
                if held and marker.exists() and held_marker_seen_at is None:
                    held_marker_seen_at = time.monotonic()
                    milestone["discovery_hold_first"] = round(held_marker_seen_at - start, 3)
                for name, path in (("live", "/healthz/live"), ("status", "/api/status"), ("metrics", "/metrics")):
                    code, body, latency = request(base, path, token)
                    counts[f"{name}_requests"] += 1
                    max_latency[name] = max(max_latency.get(name, 0.0), latency)
                    if code is None:
                        counts[f"{name}_timeouts"] += 1
                    elif code == 200:
                        counts[f"{name}_ok"] += 1
                        milestone.setdefault("api_first", round(time.monotonic() - start, 3))
                        if name == "status" and isinstance(body, dict):
                            latest_status = body
                            if held_marker_seen_at is not None:
                                held_status = body
                                held_status_latency = latency
                            catchup = body.get("catchup")
                            if isinstance(catchup, dict):
                                receipt["status_catchup_counts"] = {
                                    "cumulative_succeeded_file_count": catchup.get("cumulative_succeeded_file_count"),
                                    "cumulative_failed_file_attempts": catchup.get("cumulative_failed_file_attempts"),
                                    "cumulative_deferred_file_count": catchup.get("cumulative_deferred_file_count"),
                                    "cumulative_ingested_bytes": catchup.get("cumulative_ingested_bytes"),
                                    "cumulative_available": catchup.get("cumulative_available"),
                                    "unavailable_reason": catchup.get("cumulative_unavailable_reason"),
                                }
                            snapshot = body.get("status_snapshot")
                            if isinstance(catchup, dict):
                                phase = catchup.get("current_phase")
                                if isinstance(phase, str):
                                    phases.add(phase)
                                if phase == "discovering":
                                    milestone.setdefault("discovery_first", round(time.monotonic() - start, 3))
                                samples.append(
                                    {
                                        "elapsed_s": round(time.monotonic() - start, 3),
                                        "run_id": body.get("run_id"),
                                        "snapshot_age_s": snapshot.get("age_s") if isinstance(snapshot, dict) else None,
                                        "phase": phase,
                                        "pending": catchup.get("discovery_pending"),
                                        "last_advanced_age_s": catchup.get("discovery_last_advanced_age_s"),
                                        "inspected": catchup.get("discovery_inspected_count"),
                                        "accepted": catchup.get("discovery_accepted_count"),
                                        "rejected": catchup.get("discovery_rejected_count"),
                                    }
                                )
                        if name == "metrics" and isinstance(body, str):
                            latest_metrics = body
                            if held_marker_seen_at is not None:
                                held_metrics_latency = latency
                if (
                    held
                    and held_marker_seen_at is not None
                    and held_status is not None
                    and held_metrics_latency is not None
                    and not held_checked
                ):
                    catchup = held_status.get("catchup")
                    if (
                        not isinstance(catchup, dict)
                        or catchup.get("current_phase") != "discovering"
                        or not catchup.get("discovery_pending")
                    ):
                        receipt["outcome"] = "responsiveness_failure"
                        raise AssertionError(f"held discovery was not visible in status: {catchup!r}")
                    if not isinstance(catchup.get("discovery_age_s"), (int, float)):
                        receipt["outcome"] = "responsiveness_failure"
                        raise AssertionError("held discovery status omitted its live age")
                    if catchup.get("planned_file_count") is not None or catchup.get("eta_s") is not None:
                        receipt["outcome"] = "responsiveness_failure"
                        raise AssertionError("unmeasured discovery denominator was presented as complete")
                    snapshot = held_status.get("status_snapshot")
                    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("age_s"), (int, float)):
                        receipt["outcome"] = "responsiveness_failure"
                        raise AssertionError("held discovery status omitted snapshot age")
                    if held_status_latency is None or held_status_latency >= 2 or held_metrics_latency >= 2:
                        receipt["outcome"] = "responsiveness_failure"
                        raise AssertionError("status or metrics missed held-discovery request deadline")
                    held_checked = True
                    release.touch()
                if (
                    held
                    and held_marker_seen_at is not None
                    and not held_checked
                    and time.monotonic() - held_marker_seen_at > 25
                ):
                    receipt["outcome"] = "responsiveness_failure"
                    raise AssertionError("status and metrics did not respond during the held discovery")
                raw_count = _durable_raw_count(archive)
                if raw_count:
                    milestone.setdefault("durable_acquisition_first", round(time.monotonic() - start, 3))
                    durable_raw_count_max = max(durable_raw_count_max, raw_count)
                    receipt["durable_raw_count_max"] = durable_raw_count_max
                if counts["status_ok"] and counts["metrics_ok"] and (not held or held_checked):
                    for session_id in SESSION_IDS:
                        if session_id in verified:
                            continue
                        path = f"/api/sessions/{quote(session_id, safe='')}/messages?limit=20"
                        code, body, latency = request(base, path, token)
                        counts["read_requests"] += 1
                        max_latency["read"] = max(max_latency.get("read", 0.0), latency)
                        if code == 200 and isinstance(body, dict) and body.get("total") == 12:
                            messages = body.get("messages")
                            expected = {f"{session_id}:n:msg-{session_id.split(':', 1)[1]}-{n:03d}" for n in range(12)}
                            observed = (
                                {str(msg.get("id")) for msg in messages if isinstance(msg, dict)}
                                if isinstance(messages, list)
                                else set()
                            )
                            if observed == expected:
                                verified.add(session_id)
                                milestone.setdefault("public_read_first", round(time.monotonic() - start, 3))
                                milestone.setdefault("first_publication", round(time.monotonic() - start, 6))
                    if len(verified) == 3:
                        # Search returns message hits. The three sessions have 36
                        # matching messages, so the first page must contain all of them.
                        search_path = "/api/sessions?" + urlencode({"query": SEARCH_TOKEN, "limit": 50})
                        code, body, latency = request(base, search_path, token)
                        counts["search_requests"] += 1
                        max_latency["search"] = max(max_latency.get("search", 0.0), latency)
                        hits = body.get("hits") if isinstance(body, dict) else None
                        found = (
                            {
                                str(hit["session"].get("id"))
                                for hit in hits
                                if isinstance(hit, dict) and isinstance(hit.get("session"), dict)
                            }
                            if isinstance(hits, list)
                            else set()
                        )
                        if code == 200 and found == set(SESSION_IDS):
                            milestone["public_search_all"] = round(time.monotonic() - start, 3)
                            milestone["terminal_convergence_observed"] = round(time.monotonic() - start, 3)
                            receipt["outcome"] = "success"
                            break
                    if malformed_last and durable_raw_count_max >= 3:
                        # A deterministic parser failure on unchanged bytes is a
                        # settled refusal: the cold generation completes and
                        # publishes the two sound sessions, and the malformed
                        # third keeps a durable parse error instead of a session.
                        if SESSION_IDS[2] in verified:
                            raise AssertionError("the malformed session was published")
                        parse_error = _durable_parse_error(
                            archive, projects / FIXTURE_NESTED_SESSION
                        ) or _durable_parse_error(archive, source / FIXTURE_NESTED_SESSION)
                        if parse_error is not None and verified == set(SESSION_IDS[:2]):
                            expected_malformed_refusal = True
                            receipt["parse_refusal"] = {
                                "source": FIXTURE_NESTED_SESSION,
                                "error": parse_error[:500],
                            }
                            receipt["outcome"] = "incomplete_population"
                            raise AssertionError(
                                "two sessions published; malformed third settled as a durable parse refusal"
                            )
                time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
            else:
                if not counts["status_ok"] or not counts["metrics_ok"]:
                    receipt["outcome"] = "responsiveness_failure"
                raise AssertionError(f"cold daemon deadline: verified={sorted(verified)} outcome={receipt['outcome']}")
            if malformed_last:
                receipt["outcome"] = "incomplete_population"
                raise AssertionError("malformed control unexpectedly published all expected sessions")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        release.touch()
        if proc is not None:
            shutdown_start = time.monotonic()
            if proc.poll() is None and receipt["process_tree_rss_available"]:
                retain_tree_identities(_process_tree_rss(proc.pid))
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
                receipt["outcome"] = "shutdown_failure"
                error = "forced kill after graceful shutdown timeout"
            rss_stop.set()
            if rss_thread is not None:
                # Each pass has strict process, task and file-size caps. Wait
                # for the final bounded pass before publishing its counters.
                rss_thread.join()
            if receipt["process_tree_rss_available"]:
                survivor_deadline = time.monotonic() + 5.0
                survivors: dict[int, str] = {}
                while True:
                    survivors = {}
                    for owned_pid, start_time_ticks in observed_process_identities.items():
                        state = _owned_process_identity_state(owned_pid, start_time_ticks)
                        if state is not None:
                            survivors[owned_pid] = state
                    if not survivors or time.monotonic() >= survivor_deadline:
                        break
                    time.sleep(0.05)
                receipt["process_tree_survivor_check"] = "clear" if not survivors else "survivors"
                receipt["process_tree_survivors"] = [
                    {"pid": owned_pid, "state": state} for owned_pid, state in sorted(survivors.items())
                ]
                receipt["process_tree_survivor_check_missing_reason"] = None
                if survivors:
                    receipt["outcome"] = "shutdown_failure"
                    error = f"owned daemon descendants remained after shutdown: {sorted(survivors)}"
            if rss_samples:
                receipt["process_tree_rss_bytes"] = peak_tree_rss_bytes
                receipt["process_tree_rss_process_count_at_peak"] = peak_tree_process_count
                receipt["process_tree_rss_task_count_at_peak"] = peak_tree_task_count
                receipt["process_tree_rss_truncated_sample_count"] = rss_truncated_samples
                receipt["process_tree_rss_peak_sample_truncated"] = peak_tree_truncated
                receipt["process_tree_rss_sample_count"] = rss_samples
                receipt["process_tree_rss_missing_reason"] = None
            elif receipt["process_tree_rss_available"]:
                receipt["process_tree_rss_missing_reason"] = "no_samples"
            receipt["exit_code"] = proc.returncode
            receipt["signal"] = -proc.returncode if proc.returncode < 0 else None
            if proc.returncode != 0 and receipt["outcome"] == "success":
                receipt["outcome"] = "shutdown_failure"
                error = f"daemon exited nonzero after graceful stop: {proc.returncode}"
            milestone["shutdown_complete"] = round(time.monotonic() - start, 3)
            receipt["shutdown_s"] = round(time.monotonic() - shutdown_start, 3)
        receipt["request"] = {"counts": dict(counts), "max_latency_s": max_latency}
        receipt["phase_coverage"] = sorted(phases)
        receipt["diagnostics"] = _diagnostics(latest_metrics)
        receipt["verified_sessions"] = sorted(verified)
        receipt["last_status"] = _bounded_status(latest_status) if error else None
        receipt["last_log_records"] = _tail(event_log) + _tail(stderr) if error else []
        receipt["intake_counts"] = _intake_page_evidence(event_log)
        receipt["error"] = error
        receipt["outer_elapsed_s"] = round(time.monotonic() - start, 3)
        if measure_discovery:
            receipt["discovery_measurement"] = _read_discovery_trace(discovery_trace_path, start, milestone)
        after_digest = fixture_tree_digest(source)
        fixture = receipt["fixture"]
        assert isinstance(fixture, dict)
        fixture["sha256_after_run"] = after_digest
        fixture["unchanged_after_run"] = after_digest == digest
        if receipt["outcome"] == "success":
            intake_counts = receipt.get("intake_counts")
            if not isinstance(intake_counts, dict):
                receipt["outcome"] = "incomplete_population"
                error = "daemon published expected reads without a structured intake-page event"
            elif not isinstance(intake_counts.get("offered_bytes"), int) or intake_counts["offered_bytes"] <= 0:
                receipt["outcome"] = "incomplete_population"
                error = "daemon published expected reads without nonzero intake-page bytes"
            elif not isinstance(intake_counts.get("succeeded"), int) or intake_counts["succeeded"] <= 0:
                receipt["outcome"] = "incomplete_population"
                error = "daemon published expected reads without successful intake-page evidence"
            elif not isinstance(intake_counts.get("failed"), int) or not isinstance(intake_counts.get("deferred"), int):
                receipt["outcome"] = "incomplete_population"
                error = "daemon intake-page event omitted failed/deferred counts"
            if after_digest != digest:
                receipt["outcome"] = "incomplete_population"
                error = "sealed external source tree changed during cold qualification"
            receipt["error"] = error
        samples_path.write_text(json.dumps(list(samples), indent=2) + "\n", encoding="utf-8")
        receipt_path = emit_receipt(
            f"cold-daemon-{rejected}-{'held' if held else 'ordinary'}-{'malformed' if malformed_last else 'valid'}",
            receipt,
            env={"POLYLOGUE_MEASUREMENT_RECEIPT_DIR": str(artifacts)},
        )
        receipt["receipt_path"] = str(receipt_path)
    if error and not (
        expected_malformed_refusal and receipt["outcome"] == "incomplete_population" and receipt.get("exit_code") == 0
    ):
        raise AssertionError(f"{error}; receipt={receipt['receipt_path']}")
    return receipt


_INSTRUMENTED_BOOTSTRAP = """
import json, os, threading, time
from pathlib import Path
import polylogue.sources.live.discovery as discovery
import polylogue.operations.intake_adapters as intake_adapters

trace_path = Path(os.environ['COLD_DISCOVERY_TRACE'])
hold_discovery = os.environ.get('COLD_HOLD_DISCOVERY') == '1'
def record(name, **fields):
    row = {'name': name, 'monotonic': time.monotonic(), **fields}
    with trace_path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, sort_keys=True) + '\\n')

original_ordered = discovery._ordered_children
original_steps = intake_adapters._source_path_steps
walk_state = threading.local()
root_listing_recorded = False
hold_released = threading.Event()

def ordered_wrapper(source, directory, after, scandir=discovery.os.scandir, **kwargs):
    global root_listing_recorded
    is_first_root = (
        getattr(walk_state, 'active', False)
        and directory == source.root
        and not root_listing_recorded
    )
    if is_first_root:
        root_listing_recorded = True
        record('root_listing_start')
        if hold_discovery:
            Path(os.environ['COLD_MARKER']).touch()
            def release_when_requested():
                until = time.monotonic() + 30
                while time.monotonic() < until and not Path(os.environ['COLD_RELEASE']).exists():
                    time.sleep(.05)
                hold_released.set()
            threading.Thread(target=release_when_requested, daemon=True).start()
            hold_released.wait(timeout=30)

    actual_scandir = scandir
    if is_first_root:
        class ScandirProxy:
            def __init__(self, inner): self.inner = inner
            def __enter__(self):
                self.inner.__enter__()
                return self
            def __exit__(self, *args): return self.inner.__exit__(*args)
            def __iter__(self):
                try:
                    yield from self.inner
                finally:
                    record('root_listing_end')
        def measured_scandir(path):
            return ScandirProxy(actual_scandir(path))
        scandir = measured_scandir
    children = original_ordered(source, directory, after, scandir=scandir, **kwargs)
    if is_first_root:
        record('root_sort_end', child_count=len(children))
    return children

def steps_wrapper(*args, **kwargs):
    first = True
    was_active = getattr(walk_state, 'active', False)
    walk_state.active = True
    try:
        for item in original_steps(*args, **kwargs):
            if first:
                first = False
                record('first_yielded_entry', kind='accepted_path' if item is not None else 'step_without_accepted_path')
            yield item
    finally:
        walk_state.active = was_active

discovery._ordered_children = ordered_wrapper
intake_adapters._source_path_steps = steps_wrapper
from polylogue.daemon.commands import main
main()
"""
