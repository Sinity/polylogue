"""Prove daemon status/health HTTP responsiveness during a real raw-authority drain.

``devtools/raw_authority_scale_proof.py`` and ``devtools/raw_authority_restart_proof.py``
both call ``repair.repair_raw_materialization`` directly in-process against a synthetic
archive: there is no HTTP surface, no heartbeat, and nothing to probe. Neither proves
polylogue-hjpx.2 AC2's "daemon-health responsiveness -- status/heartbeat surfaces stay
interactive WHILE draining" clause, because neither harness runs an actual ``polylogued``
daemon.

This harness closes that gap. It (1) reuses the scale-proof corpus generator to build a
fresh, private-free synthetic raw-authority backlog, (2) starts a real ``polylogued``
subprocess pointed at that archive with ``--no-watch --no-browser-capture`` (the API
server stays on), (3) lets the daemon's own periodic
``_periodic_raw_materialization_convergence`` tick loop drain the backlog -- this harness
never calls ``repair_raw_materialization`` itself, (4) concurrently polls
``/healthz/live``, ``/healthz/ready``, and ``/api/status`` from a background thread on a
fixed interval while the drain runs, and (5) computes p50/p95/p99/max latency plus the
longest unresponsive span per endpoint. The proof fails only if an endpoint is
unresponsive (timeout/refused/5xx) for longer than a documented bound -- it records
observed latencies rather than asserting a hardcoded-tight budget, since absolute
numbers are expected to improve under the free-threading program (polylogue-xikl).

The probe/evaluate pieces (``ResponsivenessProbe``, ``summarize_endpoint``,
``evaluate_responsiveness``) are unit-tested directly against local HTTP fixtures,
including a mutation case where an endpoint never responds. The full run -- real corpus
generation, a real subprocess, a real drain -- is slow-lane (tests/integration).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from devtools.raw_authority_scale_proof import run_raw_authority_scale_proof
from polylogue.config import Config
from polylogue.storage import repair

_PROOF_SCHEMA = "polylogue.raw-authority-daemon-health-proof.v1"
_DEFAULT_PROBE_ENDPOINTS: tuple[str, ...] = ("/healthz/live", "/healthz/ready", "/api/status")
_DEFAULT_PROBE_INTERVAL_SECONDS = 0.2
_DEFAULT_PROBE_TIMEOUT_SECONDS = 2.0
_DEFAULT_MAX_UNRESPONSIVE_SECONDS = 5.0
_DEFAULT_MAX_DRAIN_SECONDS = 180.0
_DEFAULT_READINESS_TIMEOUT_SECONDS = 20.0
_DEFAULT_COMPONENTS = 64
_DEFAULT_RAWS = 64
_POST_DRAIN_SETTLE_SECONDS = 1.0


class RawAuthorityDaemonHealthProofError(RuntimeError):
    """Raised when the daemon-health responsiveness contract is not satisfied."""


@dataclass(frozen=True, slots=True)
class ProbeSample:
    """One HTTP round trip against one endpoint, timestamped since probe start."""

    endpoint: str
    sequence: int
    elapsed_since_start_s: float
    latency_ms: float | None
    ok: bool
    status_code: int | None
    error: str | None


@dataclass(frozen=True, slots=True)
class EndpointResponsiveness:
    """Latency distribution and unresponsiveness evidence for one endpoint."""

    endpoint: str
    sample_count: int
    failure_count: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    max_ms: float | None
    longest_unresponsive_span_s: float


def _percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile; ``statistics.quantiles`` needs n>=2."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    lower_weight = ordered[lower] * (upper - position)
    upper_weight = ordered[upper] * (position - lower)
    return lower_weight + upper_weight


class ResponsivenessProbe:
    """Poll fixed HTTP endpoints against one base URL on a background thread.

    Each tick issues one GET per configured endpoint, sequentially, recording
    latency and outcome. A connection refusal, timeout, or non-2xx status is
    recorded as a failed sample rather than raising -- the probe's job is to
    observe the daemon's responsiveness, never to abort the drain it is
    watching.
    """

    def __init__(
        self,
        base_url: str,
        *,
        endpoints: tuple[str, ...] = _DEFAULT_PROBE_ENDPOINTS,
        interval_seconds: float = _DEFAULT_PROBE_INTERVAL_SECONDS,
        timeout_seconds: float = _DEFAULT_PROBE_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._endpoints = endpoints
        self._interval_seconds = interval_seconds
        self._timeout_seconds = timeout_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="raw-authority-daemon-health-probe", daemon=True)
        self._samples: list[ProbeSample] = []
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._sequence = 0

    def __enter__(self) -> ResponsivenessProbe:
        self._started_at = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._interval_seconds * 4 + self._timeout_seconds + 1)

    def samples(self) -> tuple[ProbeSample, ...]:
        with self._lock:
            return tuple(self._samples)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            for endpoint in self._endpoints:
                self._sample_once(endpoint)

    def _sample_once(self, endpoint: str) -> None:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        elapsed_since_start = time.monotonic() - self._started_at
        started = time.perf_counter()
        ok = False
        status_code: int | None = None
        error: str | None = None
        try:
            request = Request(f"{self._base_url}{endpoint}", headers={"Accept": "application/json"}, method="GET")
            with urlopen(request, timeout=self._timeout_seconds) as response:
                response.read()
                status_code = response.status
            ok = 200 <= status_code < 300
        except HTTPError as exc:
            status_code = exc.code
            error = f"http_error:{exc.code}"
        except (URLError, OSError, TimeoutError) as exc:
            error = f"{type(exc).__name__}:{exc}"
        latency_ms = (time.perf_counter() - started) * 1000
        sample = ProbeSample(
            endpoint=endpoint,
            sequence=sequence,
            elapsed_since_start_s=round(elapsed_since_start, 3),
            latency_ms=round(latency_ms, 3),
            ok=ok,
            status_code=status_code,
            error=error,
        )
        with self._lock:
            self._samples.append(sample)


def _longest_unresponsive_span(samples: list[ProbeSample]) -> float:
    """Longest wall-clock stretch spanned by consecutive failed/timeout samples."""
    ordered = sorted(samples, key=lambda sample: sample.sequence)
    longest = 0.0
    run_start: float | None = None
    for sample in ordered:
        if sample.ok:
            run_start = None
            continue
        if run_start is None:
            run_start = sample.elapsed_since_start_s
        longest = max(longest, sample.elapsed_since_start_s - run_start)
    return longest


def summarize_endpoint(endpoint: str, samples: tuple[ProbeSample, ...]) -> EndpointResponsiveness:
    """Reduce one endpoint's samples to a latency distribution and failure evidence."""
    endpoint_samples = [sample for sample in samples if sample.endpoint == endpoint]
    ok_latencies = [sample.latency_ms for sample in endpoint_samples if sample.ok and sample.latency_ms is not None]
    failure_count = sum(1 for sample in endpoint_samples if not sample.ok)
    return EndpointResponsiveness(
        endpoint=endpoint,
        sample_count=len(endpoint_samples),
        failure_count=failure_count,
        p50_ms=round(_percentile(ok_latencies, 0.50), 3) if ok_latencies else None,
        p95_ms=round(_percentile(ok_latencies, 0.95), 3) if ok_latencies else None,
        p99_ms=round(_percentile(ok_latencies, 0.99), 3) if ok_latencies else None,
        max_ms=round(max(ok_latencies), 3) if ok_latencies else None,
        longest_unresponsive_span_s=round(_longest_unresponsive_span(endpoint_samples), 3),
    )


def evaluate_responsiveness(
    samples: tuple[ProbeSample, ...],
    *,
    endpoints: tuple[str, ...],
    max_unresponsive_seconds: float,
) -> tuple[EndpointResponsiveness, ...]:
    """Summarize every endpoint and fail closed on a documented unresponsive bound."""
    summaries = tuple(summarize_endpoint(endpoint, samples) for endpoint in endpoints)
    for summary in summaries:
        if summary.sample_count == 0:
            raise RawAuthorityDaemonHealthProofError(f"probe collected no samples for {summary.endpoint}")
        if summary.longest_unresponsive_span_s > max_unresponsive_seconds:
            raise RawAuthorityDaemonHealthProofError(
                f"{summary.endpoint} was unresponsive for {summary.longest_unresponsive_span_s:.2f}s, "
                f"exceeding the documented {max_unresponsive_seconds:.2f}s bound "
                f"({summary.failure_count}/{summary.sample_count} samples failed)"
            )
    return summaries


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail(path: Path, *, lines: int = 80) -> str:
    if not path.exists():
        return "<missing>"
    text = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def _wait_for_daemon_ready(
    base_url: str,
    *,
    process: subprocess.Popen[bytes],
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RawAuthorityDaemonHealthProofError(
                f"daemon exited before HTTP readiness: exit_code={process.returncode}"
            )
        try:
            with urlopen(f"{base_url}/healthz/live", timeout=1) as response:
                response.read()
            return
        except (URLError, OSError) as exc:
            last_error = exc
            time.sleep(0.05)
    raise RawAuthorityDaemonHealthProofError(
        f"daemon HTTP endpoint did not become ready within {timeout_seconds:.0f}s: "
        f"{base_url}; last_error={last_error!r}"
    )


def _wait_for_drain(
    config: Config,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 1.0,
) -> dict[str, object]:
    """Poll the read-only aggregate profile until the daemon's own loop drains it.

    ``raw_materialization_scale_profile`` is the same private-free, read-only
    aggregate query used by ``--capture-profile`` against a live archive; it is
    safe to poll concurrently with the daemon's writer.
    """
    deadline = time.monotonic() + timeout_seconds
    last_profile: dict[str, object] | None = None
    while time.monotonic() < deadline:
        profile = repair.raw_materialization_scale_profile(config)
        last_profile = profile
        if not profile.get("available", False):
            raise RawAuthorityDaemonHealthProofError(f"raw materialization profile unavailable: {profile!r}")
        if int(cast(int, profile.get("candidate_count", -1))) == 0:
            return profile
        time.sleep(poll_interval_seconds)
    raise RawAuthorityDaemonHealthProofError(
        f"raw-authority backlog did not drain through the daemon's own loop within "
        f"{timeout_seconds:.0f}s; last profile={last_profile!r}"
    )


def _terminate_daemon(daemon: subprocess.Popen[bytes]) -> None:
    if daemon.poll() is not None:
        return
    daemon.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        daemon.wait(timeout=10)
    if daemon.poll() is None:
        daemon.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            daemon.wait(timeout=10)


def run_raw_authority_daemon_health_proof(
    workdir: Path,
    *,
    components: int = _DEFAULT_COMPONENTS,
    raws: int = _DEFAULT_RAWS,
    probe_endpoints: tuple[str, ...] = _DEFAULT_PROBE_ENDPOINTS,
    probe_interval_seconds: float = _DEFAULT_PROBE_INTERVAL_SECONDS,
    probe_timeout_seconds: float = _DEFAULT_PROBE_TIMEOUT_SECONDS,
    max_unresponsive_seconds: float = _DEFAULT_MAX_UNRESPONSIVE_SECONDS,
    max_drain_seconds: float = _DEFAULT_MAX_DRAIN_SECONDS,
    readiness_timeout_seconds: float = _DEFAULT_READINESS_TIMEOUT_SECONDS,
    keep: bool = False,
    max_io_full_avg10: float | None = 2.0,
    max_memory_full_avg10: float | None = 2.0,
) -> dict[str, object]:
    """Drive a real polylogued subprocess's raw-authority drain while probing its health.

    Builds a fresh synthetic backlog (reusing the scale-proof corpus generator's
    pressure-gated generation), starts ``polylogued run --no-watch
    --no-browser-capture`` against it, and polls ``probe_endpoints`` from a
    background thread for the whole drain. The daemon's own periodic
    convergence loop performs every write; this function never calls
    ``repair_raw_materialization`` directly.
    """
    root = workdir.expanduser().resolve() / "raw-authority-daemon-health-proof"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    prepared = run_raw_authority_scale_proof(
        root / "corpus",
        components=components,
        raws=raws,
        expanded_raws=raws,
        prepare_only=True,
        keep=True,
        max_io_full_avg10=max_io_full_avg10,
        max_memory_full_avg10=max_memory_full_avg10,
    )
    archive_root = Path(cast(str, prepared["archive_root"]))
    config = Config(archive_root=archive_root, render_root=archive_root, sources=[], db_path=archive_root / "index.db")
    initial_profile = repair.raw_materialization_scale_profile(config)
    if not initial_profile.get("available") or int(cast(int, initial_profile.get("candidate_count", 0))) == 0:
        raise RawAuthorityDaemonHealthProofError(
            f"synthetic corpus left no raw-authority backlog for the daemon to drain: {initial_profile!r}"
        )

    api_port = _free_local_port()
    base_url = f"http://127.0.0.1:{api_port}"
    daemon_log_path = root / "daemon.log"
    env = os.environ.copy()
    env["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    env["POLYLOGUE_CONFIG"] = str(root / "unused-polylogue.toml")
    env["POLYLOGUE_FORCE_PLAIN"] = "1"

    with daemon_log_path.open("wb") as log:
        daemon = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from polylogue.daemon.cli import main; main()",
                "run",
                "--no-watch",
                "--no-browser-capture",
                "--no-source-catchup",
                "--api-port",
                str(api_port),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

    try:
        _wait_for_daemon_ready(base_url, process=daemon, timeout_seconds=readiness_timeout_seconds)
        drain_started = time.perf_counter()
        with ResponsivenessProbe(
            base_url,
            endpoints=probe_endpoints,
            interval_seconds=probe_interval_seconds,
            timeout_seconds=probe_timeout_seconds,
        ) as probe:
            final_profile = _wait_for_drain(config, timeout_seconds=max_drain_seconds)
            # Keep sampling briefly past the drain so the burst-to-idle tail is
            # captured, not just the burst itself.
            time.sleep(max(probe_interval_seconds * 5, _POST_DRAIN_SETTLE_SECONDS))
        drain_wall_seconds = time.perf_counter() - drain_started
        if daemon.poll() is not None:
            raise RawAuthorityDaemonHealthProofError(f"daemon exited during the drain: exit_code={daemon.returncode}")
        samples = probe.samples()
    except RawAuthorityDaemonHealthProofError as exc:
        raise RawAuthorityDaemonHealthProofError(f"{exc}\ndaemon log tail:\n{_tail(daemon_log_path)}") from exc
    finally:
        _terminate_daemon(daemon)

    try:
        summaries = evaluate_responsiveness(
            samples,
            endpoints=probe_endpoints,
            max_unresponsive_seconds=max_unresponsive_seconds,
        )
    except RawAuthorityDaemonHealthProofError as exc:
        raise RawAuthorityDaemonHealthProofError(f"{exc}\ndaemon log tail:\n{_tail(daemon_log_path)}") from exc

    report: dict[str, object] = {
        "schema": _PROOF_SCHEMA,
        "archive_root": str(archive_root),
        "requested_shape": prepared["requested_shape"],
        "achieved_shape": prepared["achieved_shape"],
        "initial_backlog_candidate_count": initial_profile["candidate_count"],
        "final_backlog_candidate_count": final_profile["candidate_count"],
        "drain_wall_seconds": round(drain_wall_seconds, 3),
        "probe": {
            "endpoints": list(probe_endpoints),
            "interval_seconds": probe_interval_seconds,
            "timeout_seconds": probe_timeout_seconds,
            "max_unresponsive_seconds_bound": max_unresponsive_seconds,
            "sample_count": len(samples),
            "endpoint_summaries": [asdict(summary) for summary in summaries],
        },
        "timeline": [asdict(sample) for sample in samples],
        "success": True,
    }
    report_path = root / "raw-authority-daemon-health-proof.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not keep:
        shutil.rmtree(root)
    return report


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path(".cache"))
    parser.add_argument("--components", type=int, default=_DEFAULT_COMPONENTS)
    parser.add_argument("--raws", type=int, default=_DEFAULT_RAWS)
    parser.add_argument("--probe-interval-s", type=float, default=_DEFAULT_PROBE_INTERVAL_SECONDS)
    parser.add_argument("--probe-timeout-s", type=float, default=_DEFAULT_PROBE_TIMEOUT_SECONDS)
    parser.add_argument("--max-unresponsive-s", type=float, default=_DEFAULT_MAX_UNRESPONSIVE_SECONDS)
    parser.add_argument("--max-drain-s", type=float, default=_DEFAULT_MAX_DRAIN_SECONDS)
    parser.add_argument("--readiness-timeout-s", type=float, default=_DEFAULT_READINESS_TIMEOUT_SECONDS)
    parser.add_argument("--max-io-full-avg10", type=float, default=2.0)
    parser.add_argument("--max-memory-full-avg10", type=float, default=2.0)
    parser.add_argument("--allow-contended-host", action="store_true")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    pressure_limit = None if args.allow_contended_host else args.max_io_full_avg10
    memory_limit = None if args.allow_contended_host else args.max_memory_full_avg10
    payload = run_raw_authority_daemon_health_proof(
        args.workdir,
        components=args.components,
        raws=args.raws,
        probe_interval_seconds=args.probe_interval_s,
        probe_timeout_seconds=args.probe_timeout_s,
        max_unresponsive_seconds=args.max_unresponsive_s,
        max_drain_seconds=args.max_drain_s,
        readiness_timeout_seconds=args.readiness_timeout_s,
        keep=args.keep,
        max_io_full_avg10=pressure_limit,
        max_memory_full_avg10=memory_limit,
    )
    out = stdout if stdout is not None else sys.stdout
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True), file=out)
    else:
        probe_payload = cast(dict[str, object], payload["probe"])
        summaries = cast(list[dict[str, object]], probe_payload["endpoint_summaries"])
        for summary in summaries:
            print(
                f"{summary['endpoint']}: p50={summary['p50_ms']}ms p95={summary['p95_ms']}ms "
                f"p99={summary['p99_ms']}ms max={summary['max_ms']}ms "
                f"failures={summary['failure_count']}/{summary['sample_count']}",
                file=out,
            )
        print(f"drain_wall_seconds={payload['drain_wall_seconds']}", file=out)
    return 0


__all__ = [
    "EndpointResponsiveness",
    "ProbeSample",
    "RawAuthorityDaemonHealthProofError",
    "ResponsivenessProbe",
    "evaluate_responsiveness",
    "main",
    "run_raw_authority_daemon_health_proof",
    "summarize_endpoint",
]


if __name__ == "__main__":
    raise SystemExit(main())
