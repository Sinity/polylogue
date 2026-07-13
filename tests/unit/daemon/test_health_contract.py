"""HTTP health-endpoint contract tests (#1224).

Pins the operator-facing contracts of the daemon health surface:

1. **Per-tier inventory** — FAST/MEDIUM/EXPENSIVE check sets are explicit.
   Adding or removing a check is a contract change visible in this file.
2. **HealthAlert envelope** — every alert carries name + tier + severity +
   message + checked_at + consecutive_failures. Severity is a closed enum.
3. **Check-level failure isolation** — a single check raising must surface
   as a typed alert in that slot, never propagate as a 500.
4. **Kubernetes probes** — `GET /healthz/live` is heartbeat-only (no DB,
   no subsystem); `GET /healthz/ready` returns 503 with a structured
   taxonomy reason when not ready.
5. **Degraded-reason taxonomy** — process-local degraded flag surfaces
   through the readiness probe with the explicit code.

In-process HTTP handler harness (mirrors test_daemon_http_security.py /
test_daemon_http_contracts.py): no real daemon, no socket listener.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

from polylogue.daemon import health as health_module
from polylogue.daemon.health import (
    HealthAlert,
    HealthSeverity,
    HealthTier,
    _run_expensive_checks,
    _run_fast_checks,
    _run_medium_checks,
    check_health,
)

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer


# ---------------------------------------------------------------------------
# Per-tier inventory — pinned contract (#1224 AC)
# ---------------------------------------------------------------------------

EXPECTED_FAST_CHECKS: frozenset[str] = frozenset(
    {
        "daemon_liveness",
        "schema_version",
        "disk_space",
        "wal_size",
        "source_availability",
        "hook_flow",
    }
)
EXPECTED_MEDIUM_CHECKS: frozenset[str] = frozenset(
    {
        "fts_readiness",
        "raw_failures",
        "stale_ingest_attempts",
        "insight_freshness",
        "repeated_stage_failures",
    }
)
EXPECTED_EXPENSIVE_CHECKS: frozenset[str] = frozenset(
    {
        "db_integrity",
        "blob_integrity",
        "embedding_coverage",
    }
)

# The closed severity vocabulary that operators consume.
EXPECTED_SEVERITIES: frozenset[str] = frozenset({"ok", "warning", "error", "critical"})

# Degraded-reason taxonomy returned by `/healthz/ready` when not ready.
# Adding a code is a contract change — extend this set in the same PR.
EXPECTED_READINESS_REASONS: frozenset[str] = frozenset(
    {
        "schema_version_mismatch",
        "critical_check_failed",
        "fts_not_fresh",
        "probe_error",
    }
)


def _seed_ready_message_fts(index_db: Path) -> None:
    index_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(index_db) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS fts_freshness_state (
                surface TEXT PRIMARY KEY,
                state TEXT NOT NULL CHECK (state IN ('ready', 'stale', 'unknown')),
                checked_at TEXT NOT NULL,
                source_rows INTEGER NOT NULL DEFAULT 0,
                indexed_rows INTEGER NOT NULL DEFAULT 0,
                missing_rows INTEGER NOT NULL DEFAULT 0,
                excess_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0,
                detail TEXT
            ) STRICT;
            INSERT OR REPLACE INTO fts_freshness_state
            VALUES ('messages_fts', 'ready', '2026-05-24T00:00:00+00:00', 0, 0, 0, 0, 0, 'ready');
            """
        )


# ---------------------------------------------------------------------------
# HTTP handler harness
# ---------------------------------------------------------------------------


class _MockServer:
    auth_token = ""
    api_host = "127.0.0.1"
    archive_query_executor = ThreadPoolExecutor(max_workers=1)
    archive_query_admission = threading.BoundedSemaphore(64)  # generous: not under test


class _MockHeaders:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self._headers = headers or {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._headers.get(key, default)


def _make_handler(method: str, path: str, *, body: bytes = b"") -> DaemonAPIHandler:
    from polylogue.daemon.http import DaemonAPIHandler

    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.server = cast("DaemonAPIHTTPServer", _MockServer())
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.command = method
    handler.requestline = f"{method} {path} HTTP/1.1"
    headers: dict[str, str] = {"Content-Length": str(len(body))}
    handler.headers = cast("Message[str, str]", _MockHeaders(headers))
    handler.rfile = BytesIO(body)
    handler.wfile = BytesIO()
    return handler


def _capture_responses(handler: DaemonAPIHandler) -> tuple[MagicMock, MagicMock]:
    send_error = MagicMock()
    send_json = MagicMock()
    handler._send_error = send_error  # type: ignore[method-assign]
    handler._send_json = send_json  # type: ignore[method-assign]
    return send_error, send_json


# ---------------------------------------------------------------------------
# 1. Per-tier inventory contract
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestTierInventoryContract:
    """Each tier exposes exactly the documented check set."""

    def test_fast_tier_inventory_pinned(self) -> None:
        names = frozenset(alert.check_name for alert in _run_fast_checks())
        missing = EXPECTED_FAST_CHECKS - names
        extra = names - EXPECTED_FAST_CHECKS
        assert not missing, f"FAST tier missing checks: {sorted(missing)}"
        assert not extra, f"FAST tier has undocumented checks: {sorted(extra)}"

    def test_medium_tier_inventory_pinned(self) -> None:
        names = frozenset(alert.check_name for alert in _run_medium_checks())
        missing = EXPECTED_MEDIUM_CHECKS - names
        extra = names - EXPECTED_MEDIUM_CHECKS
        assert not missing, f"MEDIUM tier missing checks: {sorted(missing)}"
        assert not extra, f"MEDIUM tier has undocumented checks: {sorted(extra)}"

    @pytest.mark.slow
    def test_expensive_tier_inventory_pinned(self) -> None:
        names = frozenset(alert.check_name for alert in _run_expensive_checks())
        missing = EXPECTED_EXPENSIVE_CHECKS - names
        extra = names - EXPECTED_EXPENSIVE_CHECKS
        assert not missing, f"EXPENSIVE tier missing checks: {sorted(missing)}"
        assert not extra, f"EXPENSIVE tier has undocumented checks: {sorted(extra)}"


# ---------------------------------------------------------------------------
# 2. HealthAlert envelope contract
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestHealthAlertEnvelopeContract:
    """Every alert carries the documented fields with typed severity."""

    def test_alert_envelope_fields_present(self) -> None:
        for alert in _run_fast_checks() + _run_medium_checks():
            # name + tier + severity + message + checked_at are required.
            assert alert.check_name, f"alert missing name: {alert}"
            assert isinstance(alert.tier, HealthTier)
            assert isinstance(alert.severity, HealthSeverity)
            assert alert.severity.value in EXPECTED_SEVERITIES
            assert isinstance(alert.message, str)
            assert alert.message
            assert isinstance(alert.checked_at, str)
            assert alert.checked_at
            assert isinstance(alert.consecutive_failures, int)
            assert alert.consecutive_failures >= 0

    def test_alert_serializes_to_stable_json_shape(self) -> None:
        alert = HealthAlert(
            check_name="example",
            tier=HealthTier.FAST,
            severity=HealthSeverity.WARNING,
            message="example warning",
            checked_at="2026-05-18T00:00:00+00:00",
            consecutive_failures=2,
        )
        payload = alert.model_dump(mode="json")
        # Exact field set — adding a field is a contract change.
        assert frozenset(payload.keys()) == frozenset(
            {"check_name", "tier", "severity", "message", "checked_at", "consecutive_failures"}
        )
        # Severity and tier serialize as the enum value, not a class repr.
        assert payload["severity"] == "warning"
        assert payload["tier"] == "fast"


# ---------------------------------------------------------------------------
# 3. Check-level failure isolation contract
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestCheckFailureIsolationContract:
    """A single check raising surfaces as a typed alert, never as a 500."""

    @pytest.fixture(autouse=True)
    def _reset_failures(self) -> Iterator[None]:
        health_module._failure_counts.clear()
        yield
        health_module._failure_counts.clear()

    def test_individual_check_exception_becomes_error_alert(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _explode(_path: str) -> object:
            raise RuntimeError("simulated subsystem failure")

        # The disk-space check wraps OSError but a foreign RuntimeError from
        # an internal helper should still produce a typed alert (the check
        # bodies catch broad `Exception`).
        from polylogue.daemon.health import _check_disk_space_fast

        monkeypatch.setattr("polylogue.daemon.health.os.statvfs", _explode)
        alert = _check_disk_space_fast()
        assert alert.check_name == "disk_space"
        assert alert.severity == HealthSeverity.ERROR
        assert "disk check failed" in alert.message
        # The exception text never leaks as a free-form string outside the message.
        assert isinstance(alert.checked_at, str)

    def test_check_health_aggregate_survives_individual_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When one check raises, `check_health()` still returns a DaemonHealth.

        The aggregator must never propagate a per-check exception; the AC is
        "when a check raises, the endpoint returns a typed error in that
        check's slot — not a 500".
        """
        bad = HealthAlert(
            check_name="schema_version",
            tier=HealthTier.FAST,
            severity=HealthSeverity.ERROR,
            message="schema check failed: forced",
            checked_at="2026-05-18T00:00:00+00:00",
            consecutive_failures=1,
        )
        ok = HealthAlert(
            check_name="daemon_liveness",
            tier=HealthTier.FAST,
            severity=HealthSeverity.OK,
            message="alive",
            checked_at="2026-05-18T00:00:00+00:00",
        )
        monkeypatch.setattr(health_module, "_run_fast_checks", lambda: [bad, ok])
        monkeypatch.setattr(health_module, "_run_medium_checks", lambda: [])

        result = check_health(tiers={HealthTier.FAST})
        # Both alerts are present — failed check did NOT bring down the aggregate.
        names = [a.check_name for a in result.alerts]
        assert "schema_version" in names
        assert "daemon_liveness" in names
        assert result.overall_status == HealthSeverity.ERROR


# ---------------------------------------------------------------------------
# 4. Kubernetes-style probe contracts
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestLivenessProbeContract:
    """`GET /healthz/live` is a heartbeat-only probe."""

    def test_liveness_returns_200_with_documented_envelope(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        handler = _make_handler("GET", "/healthz/live")
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        send_json.assert_called_once()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert isinstance(payload, dict)
        assert payload["status"] == "alive"
        assert isinstance(payload["pid"], int)
        assert payload["pid"] > 0
        assert isinstance(payload["uptime_s"], float)
        assert payload["uptime_s"] >= 0
        assert isinstance(payload["started_at"], float)

    def test_liveness_uses_shared_process_start_anchor(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("polylogue.daemon.healthz.started_at_wall", lambda: 1234.5)
        monkeypatch.setattr("polylogue.daemon.healthz.uptime_seconds", lambda: 42.25)

        handler = _make_handler("GET", "/healthz/live")
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["started_at"] == 1234.5
        assert payload["uptime_s"] == 42.25

    def test_liveness_reports_in_process_heartbeat_age(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("polylogue.daemon.lifecycle.process_heartbeat_age_seconds", lambda: 3.25)

        handler = _make_handler("GET", "/healthz/live")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        _, payload = send_json.call_args.args
        assert payload["heartbeat_age_s"] == 3.25

    def test_liveness_does_not_query_health_subsystem(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Liveness MUST NOT call `check_health()` or any subsystem probe.

        A liveness probe that blocks on a stuck subsystem causes k8s to
        kill the pod — the exact failure mode liveness must guard against.
        """
        sentinel = MagicMock(side_effect=AssertionError("check_health called"))
        monkeypatch.setattr(health_module, "check_health", sentinel)

        handler = _make_handler("GET", "/healthz/live")
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        # Probe succeeds without ever invoking health checks.
        sentinel.assert_not_called()
        status, _ = send_json.call_args.args
        assert status == HTTPStatus.OK

    def test_liveness_is_unauthenticated(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """Probes don't carry credentials — k8s/docker can't supply tokens.

        Even when auth_token is configured, /healthz/live must answer
        without an Authorization header.
        """
        handler = _make_handler("GET", "/healthz/live")
        # Force an auth token on the server — probe should bypass auth check.
        handler.server.auth_token = "secret-token"
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        send_json.assert_called_once()
        status, _ = send_json.call_args.args
        assert status == HTTPStatus.OK


@pytest.mark.contract
class TestReadinessProbeContract:
    """`GET /healthz/ready` reports traffic-routing readiness."""

    @pytest.fixture(autouse=True)
    def _reset(self) -> Iterator[None]:
        from polylogue.core.degraded import clear_degraded

        clear_degraded()
        health_module._failure_counts.clear()
        yield
        clear_degraded()
        health_module._failure_counts.clear()

    def _patch_healthy_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force every FAST check OK so readiness returns ready."""
        ok_alerts = [
            HealthAlert(
                check_name=name,
                tier=HealthTier.FAST,
                severity=HealthSeverity.OK,
                message=f"{name} ok",
                checked_at="2026-05-18T00:00:00+00:00",
            )
            for name in EXPECTED_FAST_CHECKS
        ]
        monkeypatch.setattr(health_module, "_run_fast_checks", lambda: ok_alerts)
        monkeypatch.setattr(health_module, "_run_medium_checks", lambda: [])
        monkeypatch.setattr(health_module, "_run_expensive_checks", lambda: [])

    def test_readiness_returns_200_when_healthy(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._patch_healthy_fast(monkeypatch)
        _seed_ready_message_fts(workspace_env["archive_root"] / "index.db")
        handler = _make_handler("GET", "/healthz/ready")
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["status"] == "ready"
        assert payload["overall"] == "ok"
        assert isinstance(payload["checks"], list)
        names = {check["name"] for check in payload["checks"]}
        assert names == EXPECTED_FAST_CHECKS

    def test_readiness_uses_bounded_fts_freshness_not_exact_scan(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._patch_healthy_fast(monkeypatch)
        # The readiness probe reads the index.db messages_fts surface; a
        # fresh, trigger-backed surface is ready and the bounded path must not
        # fall back to an exact ``fts_invariant_snapshot_sync`` scan.
        index_db = workspace_env["archive_root"] / "index.db"
        _seed_ready_message_fts(index_db)
        monkeypatch.setattr(
            "polylogue.daemon.fts_status.fts_invariant_snapshot_sync",
            lambda _conn: (_ for _ in ()).throw(AssertionError("readiness must not run exact FTS scans")),
        )

        handler = _make_handler("GET", "/healthz/ready")
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["status"] == "ready"
        assert payload["fts"]["coverage_exact"] is False

    def test_readiness_returns_503_on_critical_check(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        crit = HealthAlert(
            check_name="disk_space",
            tier=HealthTier.FAST,
            severity=HealthSeverity.CRITICAL,
            message="disk full",
            checked_at="2026-05-18T00:00:00+00:00",
            consecutive_failures=1,
        )
        ok = HealthAlert(
            check_name="schema_version",
            tier=HealthTier.FAST,
            severity=HealthSeverity.OK,
            message="schema ok",
            checked_at="2026-05-18T00:00:00+00:00",
        )
        monkeypatch.setattr(health_module, "_run_fast_checks", lambda: [crit, ok])
        monkeypatch.setattr(health_module, "_run_medium_checks", lambda: [])

        handler = _make_handler("GET", "/healthz/ready")
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.SERVICE_UNAVAILABLE
        assert payload["status"] == "not_ready"
        assert payload["reason"] == "critical_check_failed"
        assert payload["reason"] in EXPECTED_READINESS_REASONS

    def test_readiness_returns_503_when_fts_not_fresh(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import sqlite3

        self._patch_healthy_fast(monkeypatch)
        # index.db with a block source but no fresh messages_fts state means the
        # message FTS invariant is not ready, so the probe must return 503.
        index_db = workspace_env["archive_root"] / "index.db"
        index_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(index_db) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS fts_freshness_state (
                    surface TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    checked_at TEXT NOT NULL
                );
                DELETE FROM fts_freshness_state WHERE surface = 'messages_fts';
                INSERT INTO blocks (
                    message_id, session_id, position, block_type, text
                ) VALUES ('message-1', 'session-1', 0, 'text', 'missing from fts');
                """
            )

        handler = _make_handler("GET", "/healthz/ready")
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.SERVICE_UNAVAILABLE
        assert payload["status"] == "not_ready"
        assert payload["reason"] == "fts_not_fresh"
        assert payload["reason"] in EXPECTED_READINESS_REASONS

    def test_readiness_returns_503_on_schema_mismatch(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bad_schema = HealthAlert(
            check_name="schema_version",
            tier=HealthTier.FAST,
            severity=HealthSeverity.CRITICAL,
            message="schema v2 is not runtime v3",
            checked_at="2026-05-18T00:00:00+00:00",
            consecutive_failures=1,
        )
        monkeypatch.setattr(health_module, "_run_fast_checks", lambda: [bad_schema])
        monkeypatch.setattr(health_module, "_run_medium_checks", lambda: [])

        handler = _make_handler("GET", "/healthz/ready")
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.SERVICE_UNAVAILABLE
        assert payload["reason"] == "schema_version_mismatch"
        assert payload["reason"] in EXPECTED_READINESS_REASONS

    def test_readiness_surfaces_degraded_reason_taxonomy(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """When the process-local degraded flag is set, /healthz/ready
        returns 503 with the explicit DegradedReason.code.

        This is the documented taxonomy field consumers (operator
        dashboards, k8s) switch on.
        """
        from polylogue.core.degraded import DegradedReason, set_degraded

        set_degraded(
            DegradedReason(
                code="schema_version_mismatch",
                message="binary v3 vs db v2",
                detail={"runtime": 3, "db": 2},
            )
        )

        handler = _make_handler("GET", "/healthz/ready")
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.SERVICE_UNAVAILABLE
        assert payload["status"] == "not_ready"
        assert payload["reason"] == "schema_version_mismatch"
        assert payload["reason"] in EXPECTED_READINESS_REASONS
        assert payload["message"] == "binary v3 vs db v2"
        assert payload["detail"] == {"runtime": 3, "db": 2}

    def test_readiness_probe_error_returns_503_not_500(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Internal failure in the probe path produces a structured 503,
        never a 500 with a raw traceback.
        """

        def _explode(**_kwargs: object) -> object:
            raise RuntimeError("forced probe failure")

        monkeypatch.setattr(health_module, "check_health", _explode)

        handler = _make_handler("GET", "/healthz/ready")
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.SERVICE_UNAVAILABLE
        assert payload["status"] == "not_ready"
        assert payload["reason"] == "probe_error"

    def test_readiness_is_unauthenticated(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._patch_healthy_fast(monkeypatch)
        handler = _make_handler("GET", "/healthz/ready")
        handler.server.auth_token = "secret-token"
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        send_json.assert_called_once()


# ---------------------------------------------------------------------------
# 5. JSON-shape sanity for downstream consumers (#1236 OCI HEALTHCHECK)
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_probe_payloads_are_orjson_serializable(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both probes return payloads orjson can serialize without coercion.

    The daemon serializes via orjson; non-trivial nested types (Path,
    Enum, datetime) would raise at write time and leak as 500.
    """
    import orjson

    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    # Healthy ready
    monkeypatch.setattr(health_module, "_run_fast_checks", lambda: [])
    monkeypatch.setattr(health_module, "_run_medium_checks", lambda: [])
    handler = _make_handler("GET", "/healthz/ready")
    _, send_json = _capture_responses(handler)
    handler.do_GET()
    _, payload = send_json.call_args.args
    orjson.dumps(payload)

    # Degraded ready
    set_degraded(DegradedReason(code="x", message="y", detail={"a": 1}))
    try:
        handler2 = _make_handler("GET", "/healthz/ready")
        _, send_json2 = _capture_responses(handler2)
        handler2.do_GET()
        _, payload2 = send_json2.call_args.args
        orjson.dumps(payload2)
    finally:
        clear_degraded()

    # Live
    handler3 = _make_handler("GET", "/healthz/live")
    _, send_json3 = _capture_responses(handler3)
    handler3.do_GET()
    _, payload3 = send_json3.call_args.args
    orjson.dumps(payload3)


# Compile-time guard: keep `json` import referenced so future changes can
# inspect the wire format directly if needed.
_ = json
