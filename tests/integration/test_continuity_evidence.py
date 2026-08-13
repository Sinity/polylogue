"""End-to-end proof of the continuity replay artifact.

Runs the full wiring: the real continuity scenario catalog over real MCP stdio
JSON-RPC against a freshly seeded synthetic archive, cross-checked against the
real query-discovery catalog and combined into one JSON artifact.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from devtools import continuity_replay
from devtools.continuity_evidence import main, run_continuity_evidence
from tests.infra.continuity import continuity_catalog_path, load_continuity_catalog, seed_continuity_archive


@pytest.mark.asyncio
async def test_continuity_evidence_end_to_end_synthetic_lane() -> None:
    report = await run_continuity_evidence(redact=False)

    assert report["schema_version"] == 3
    assert report["live_archive"] is False

    continuity = report["continuity"]
    assert isinstance(continuity, dict)
    assert continuity["scenario_count"] == 8
    assert continuity["status"] == "pass"

    discovery = report["discovery_coverage"]
    assert isinstance(discovery, dict)
    assert discovery["status"] == "pass"
    assert discovery["gaps"] == []

    assert report["status"] == "pass"

    # Full report round-trips through JSON (it must be a valid standalone artifact).
    json.dumps(report)


@pytest.mark.asyncio
async def test_supplied_archive_uses_matching_catalog_without_runtime_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = tmp_path / "archive"
    seed_continuity_archive(archive_root, catalog=load_continuity_catalog())
    tier_digests_before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(archive_root.glob("*.db"))
    }
    assert tier_digests_before

    requests: list[str] = []

    class FakeDaemonHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            requests.append(self.path)
            response = json.dumps({"call_id": payload.get("call_id")}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    daemon = ThreadingHTTPServer(("127.0.0.1", 0), FakeDaemonHandler)
    daemon_thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    daemon_thread.start()
    daemon_url = f"http://127.0.0.1:{daemon.server_port}"
    monkeypatch.setattr(continuity_replay, "_CONTINUITY_DAEMON_SINK_URL", daemon_url)
    monkeypatch.setenv("POLYLOGUE_DAEMON_URL", daemon_url)
    monkeypatch.setenv("POLYLOGUE_API_AUTH_TOKEN", "inherited-live-token-must-not-cross")
    monkeypatch.setenv("POLYLOGUE_MCP_WRITE_ENABLED", "1")
    monkeypatch.setenv("POLYLOGUE_MCP_JUDGE_ENABLED", "1")
    monkeypatch.setenv("POLYLOGUE_MCP_MAINTENANCE_ENABLED", "1")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "inherited-live-data"))

    try:
        report = await run_continuity_evidence(
            archive_root=archive_root,
            catalog_path=continuity_catalog_path(),
            scenario_names=("resume",),
            redact=False,
        )
    finally:
        daemon.shutdown()
        daemon.server_close()
        daemon_thread.join(timeout=5)

    assert report["status"] == "pass"
    assert report["live_archive"] is True
    assert isinstance(report["catalog_sha256"], str)
    assert requests == []
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(archive_root.glob("*.db"))
    } == tier_digests_before
    assert not (archive_root / ".continuity-runtime").exists()


def test_main_cli_writes_json_output_and_returns_pass_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "continuity-evidence.json"

    async def fake_replay(**_kwargs: object) -> dict[str, object]:
        return {"schema_version": 3, "status": "pass"}

    monkeypatch.setattr("devtools.continuity_evidence.run_continuity_evidence", fake_replay)

    exit_code = main(
        [
            "--no-redact",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["status"] == "pass"
