from __future__ import annotations

import http.server
import threading
from pathlib import Path

import pytest

from devtools.command_catalog import COMMANDS
from devtools.raw_authority_daemon_health_proof import (
    ProbeSample,
    RawAuthorityDaemonHealthProofError,
    ResponsivenessProbe,
    _percentile,
    evaluate_responsiveness,
    main,
    summarize_endpoint,
)


class _FixedHandler(http.server.BaseHTTPRequestHandler):
    """Minimal handler: every path answers 200 immediately."""

    def do_GET(self) -> None:
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:  # silence default stderr logging
        pass


@pytest.fixture
def healthy_server() -> object:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FixedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _sample(endpoint: str, sequence: int, elapsed: float, *, ok: bool, latency_ms: float | None = 1.0) -> ProbeSample:
    return ProbeSample(
        endpoint=endpoint,
        sequence=sequence,
        elapsed_since_start_s=elapsed,
        latency_ms=latency_ms,
        ok=ok,
        status_code=200 if ok else None,
        error=None if ok else "boom",
    )


def test_percentile_matches_known_distribution() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile(values, 0.0) == 10.0
    assert _percentile(values, 1.0) == 50.0
    assert _percentile(values, 0.5) == 30.0


def test_percentile_single_value_is_that_value() -> None:
    assert _percentile([7.5], 0.99) == 7.5


def test_responsiveness_probe_records_latency_for_healthy_endpoint(
    healthy_server: http.server.ThreadingHTTPServer,
) -> None:
    base_url = f"http://127.0.0.1:{healthy_server.server_address[1]}"
    with ResponsivenessProbe(base_url, endpoints=("/ok",), interval_seconds=0.02, timeout_seconds=1.0) as probe:
        import time

        time.sleep(0.3)
    samples = probe.samples()

    assert len(samples) >= 5
    assert all(sample.ok for sample in samples)
    assert all(sample.latency_ms is not None and sample.latency_ms >= 0 for sample in samples)

    (summary,) = evaluate_responsiveness(samples, endpoints=("/ok",), max_unresponsive_seconds=5.0)
    assert summary.failure_count == 0
    assert summary.sample_count == len(samples)
    assert summary.p50_ms is not None
    assert summary.p95_ms is not None
    assert summary.p99_ms is not None
    assert summary.max_ms is not None
    assert summary.longest_unresponsive_span_s == 0.0


def test_responsiveness_probe_records_connection_refused_as_failure() -> None:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        dead_port = sock.getsockname()[1]
    # Socket closed on exit; nothing listens on dead_port.
    base_url = f"http://127.0.0.1:{dead_port}"

    with ResponsivenessProbe(base_url, endpoints=("/dead",), interval_seconds=0.02, timeout_seconds=0.5) as probe:
        import time

        time.sleep(0.2)
    samples = probe.samples()

    assert samples
    assert all(not sample.ok for sample in samples)
    assert all(sample.error is not None for sample in samples)


def test_evaluate_responsiveness_fails_when_endpoint_never_responds() -> None:
    """Mutation case: a persistently unresponsive endpoint must fail the proof."""
    samples = tuple(
        _sample("/wedged", sequence, elapsed=elapsed, ok=False, latency_ms=None)
        for sequence, elapsed in enumerate([0.0, 0.3, 0.6, 0.9, 1.2, 1.5], start=1)
    )

    with pytest.raises(RawAuthorityDaemonHealthProofError, match=r"/wedged.*unresponsive"):
        evaluate_responsiveness(samples, endpoints=("/wedged",), max_unresponsive_seconds=0.5)


def test_evaluate_responsiveness_tolerates_a_brief_blip_within_bound() -> None:
    samples = (
        _sample("/flaky", 1, 0.0, ok=True),
        _sample("/flaky", 2, 0.2, ok=False, latency_ms=None),
        _sample("/flaky", 3, 0.4, ok=False, latency_ms=None),
        _sample("/flaky", 4, 0.6, ok=True),
    )

    (summary,) = evaluate_responsiveness(samples, endpoints=("/flaky",), max_unresponsive_seconds=1.0)
    assert summary.longest_unresponsive_span_s == pytest.approx(0.2)
    assert summary.failure_count == 2


def test_evaluate_responsiveness_rejects_an_endpoint_with_no_samples() -> None:
    samples = (_sample("/other", 1, 0.0, ok=True),)

    with pytest.raises(RawAuthorityDaemonHealthProofError, match="no samples for /missing"):
        evaluate_responsiveness(samples, endpoints=("/missing",), max_unresponsive_seconds=5.0)


def test_summarize_endpoint_ignores_other_endpoints_samples() -> None:
    samples = (
        _sample("/a", 1, 0.0, ok=True, latency_ms=10.0),
        _sample("/b", 2, 0.1, ok=True, latency_ms=999.0),
        _sample("/a", 3, 0.2, ok=True, latency_ms=20.0),
    )

    summary = summarize_endpoint("/a", samples)
    assert summary.sample_count == 2
    assert summary.max_ms == 20.0


def test_cli_forwards_arguments_and_catalog_entry_matches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def record_call(workdir: Path, **kwargs: object) -> dict[str, object]:
        captured["workdir"] = workdir
        captured.update(kwargs)
        return {
            "drain_wall_seconds": 1.23,
            "probe": {"endpoint_summaries": []},
        }

    monkeypatch.setattr(
        "devtools.raw_authority_daemon_health_proof.run_raw_authority_daemon_health_proof",
        record_call,
    )

    assert main(["--workdir", str(tmp_path), "--components", "8", "--raws", "8"]) == 0
    assert captured["workdir"] == tmp_path
    assert captured["components"] == 8
    assert captured["raws"] == 8
    assert captured["max_io_full_avg10"] == 2.0
    assert captured["max_memory_full_avg10"] == 2.0

    command = COMMANDS["workspace raw-authority-daemon-health-proof"]
    assert command.module == "devtools.raw_authority_daemon_health_proof"
    assert command.use_when is not None
    assert "daemon-health responsiveness" in command.use_when


def test_cli_allow_contended_host_disables_pressure_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def record_call(workdir: Path, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"drain_wall_seconds": 0.1, "probe": {"endpoint_summaries": []}}

    monkeypatch.setattr(
        "devtools.raw_authority_daemon_health_proof.run_raw_authority_daemon_health_proof",
        record_call,
    )

    assert main(["--workdir", str(tmp_path), "--allow-contended-host"]) == 0
    assert captured["max_io_full_avg10"] is None
    assert captured["max_memory_full_avg10"] is None


def test_cli_json_output_round_trips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {
        "drain_wall_seconds": 4.5,
        "probe": {
            "endpoint_summaries": [
                {
                    "endpoint": "/healthz/live",
                    "p50_ms": 1.0,
                    "p95_ms": 2.0,
                    "p99_ms": 3.0,
                    "max_ms": 4.0,
                    "failure_count": 0,
                    "sample_count": 10,
                }
            ]
        },
    }
    monkeypatch.setattr(
        "devtools.raw_authority_daemon_health_proof.run_raw_authority_daemon_health_proof",
        lambda *_args, **_kwargs: payload,
    )

    assert main(["--workdir", str(tmp_path), "--json"]) == 0
    out = capsys.readouterr().out
    assert "/healthz/live" in out
    assert "4.5" in out or "drain_wall_seconds" in out
