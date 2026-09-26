"""Private, bounded observation of a real cold ``polylogued run`` process."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from devtools.measurement_receipts import emit_receipt

SESSION_IDS = tuple(f"claude-code-session:ccccd500-0000-0000-0000-{i:012d}" for i in range(3))
SEARCH_TOKEN = "coldqualificationuniquetoken"
DEADLINE_S = 120.0


def write_fixture(root: Path, *, rejected: int, malformed_last: bool = False) -> str:
    root.mkdir(parents=True)
    digest = hashlib.sha256()
    for i in range(rejected):
        path = root / f"a-rejected-{i:05d}.txt"
        data = b"synthetic rejected entry\n"
        path.write_bytes(data)
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + data)
    for i, session_id in enumerate(SESSION_IDS):
        path = (root / "nested" if i == 2 else root) / f"z-session-{i}.jsonl"
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
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + data)
    return digest.hexdigest()


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


def _unpublished_candidate_session_count(archive: Path) -> int | None:
    """Observe cold-build work without mistaking its inactive tier for publication."""
    counts: list[int] = []
    for db in (archive / ".index-generations").glob("*/index.db"):
        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=0.1) as conn:
                counts.append(int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]))
        except sqlite3.Error:
            continue
    return max(counts) if counts else None


def qualify(
    *,
    archive: Path,
    source: Path,
    artifacts: Path,
    rejected: int,
    digest: str,
    malformed_last: bool = False,
    held: bool = False,
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
    marker = artifacts / "discovery-held"
    release = artifacts / "release-discovery"
    config = artifacts / "absent-private-config.toml"
    candidate_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update(
        {
            "POLYLOGUE_ARCHIVE_ROOT": str(archive),
            "POLYLOGUE_CONFIG": str(config),
            "POLYLOGUE_SITE_CONFIG": "",
            "POLYLOGUE_LOG_FORMAT": "json",
            "POLYLOGUE_LOG_FILE": str(event_log),
            "PYTHONPATH": str(candidate_root),
        }
    )
    for key in tuple(env):
        if key.startswith("POLYLOGUE_EMBEDDING"):
            env.pop(key)
    command = [sys.executable]
    if held:
        env.update({"COLD_MARKER": str(marker), "COLD_RELEASE": str(release)})
        command += ["-c", _HELD_BOOTSTRAP]
    else:
        command += ["-c", "from polylogue.daemon.cli import main; main()"]
    command += [
        "run",
        "--root",
        str(source),
        "--no-default-sources",
        "--no-browser-capture",
        "--api-port",
        str(port),
        "--api-auth-token",
        token,
    ]
    receipt: dict[str, object] = {
        "format": "polylogue.daemon-cold-qualification.v1",
        "candidate": {
            "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=candidate_root, text=True).strip(),
            "import_root": str(candidate_root),
            "python": sys.executable,
            "version": platform.python_version(),
        },
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
        "artifacts": {"stderr": str(stderr), "events": str(event_log), "samples": str(samples_path)},
        "outcome": "setup_failure",
        "milestones_upper_bound_s": {},
        "request": {},
        "phase_coverage": [],
        "internal_intervals": None,
        "internal_intervals_missing_reason": "no_owner_interval_evidence",
        "process_tree_rss_bytes": None,
        "process_tree_rss_missing_reason": "no_owned_tree_sampler",
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
    error: str | None = None
    try:
        with stderr.open("wb") as stream:
            proc = subprocess.Popen(command, cwd=candidate_root, env=env, stdout=stream, stderr=subprocess.STDOUT)
            receipt["pid"] = proc.pid
            receipt["outcome"] = "incomplete_population"
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
                        parse_error = _durable_parse_error(archive, source / "nested" / "z-session-2.jsonl")
                        candidate_sessions = _unpublished_candidate_session_count(archive)
                        catchup = latest_status.get("catchup") if isinstance(latest_status, dict) else None
                        failed_count = (
                            catchup.get("cumulative_failed_file_attempts") if isinstance(catchup, dict) else None
                        )
                        if (
                            parse_error is not None
                            and candidate_sessions == 2
                            and isinstance(failed_count, int)
                            and failed_count > 0
                        ):
                            if verified:
                                raise AssertionError(
                                    f"incomplete cold generation published partial sessions: {sorted(verified)}"
                                )
                            expected_malformed_refusal = True
                            receipt["parse_refusal"] = {
                                "source": "nested/z-session-2.jsonl",
                                "error": parse_error[:500],
                            }
                            receipt["candidate_sessions_unpublished"] = candidate_sessions
                            receipt["outcome"] = "incomplete_population"
                            raise AssertionError(
                                "two sessions prepared in an inactive generation; malformed third has a durable parse error"
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
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
                receipt["outcome"] = "shutdown_failure"
                error = "forced kill after graceful shutdown timeout"
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
        receipt["error"] = error
        receipt["outer_elapsed_s"] = round(time.monotonic() - start, 3)
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


_HELD_BOOTSTRAP = """
import os, threading, time
from pathlib import Path
import polylogue.sources.live.discovery as discovery
original = discovery._ordered_children
held = threading.Event()
once = threading.Event()
def wrapped(*args, **kwargs):
    if kwargs.get('on_inspected') is not None and not once.is_set():
        once.set()
        Path(os.environ['COLD_MARKER']).touch()
        def release_when_requested():
            until = time.monotonic() + 30
            while time.monotonic() < until and not Path(os.environ['COLD_RELEASE']).exists():
                time.sleep(.05)
            held.set()
        threading.Thread(target=release_when_requested, daemon=True).start()
        held.wait(timeout=30)
    return original(*args, **kwargs)
discovery._ordered_children = wrapped
from polylogue.daemon.cli import main
main()
"""
