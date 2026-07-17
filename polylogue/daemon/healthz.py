"""Kubernetes-style probe endpoints for the daemon HTTP API (#1224).

Two probes are exposed under `/healthz/` and routed unauthenticated from
``polylogue.daemon.http``:

- ``GET /healthz/live`` — liveness. Heartbeat only; never touches DB or
  subsystems. The HTTP handler returning at all is the k8s liveness
  signal; the body is a small payload for human inspection.
- ``GET /healthz/ready`` — readiness. Bounded to FAST tier health checks
  (sub-1s budget) plus the process-local degraded flag. Returns 503 with
  a structured ``reason`` from a closed taxonomy when not ready.

Both probes are unauthenticated by convention. k8s/docker/systemd
healthchecks cannot supply credentials, and the probes leak only
booleans plus structured reason codes — no archive data, no environment.

The handlers receive a typed ``ProbeResponder`` so the call sites in
``http.py`` stay tiny and so the handler internals are testable without
the full ``BaseHTTPRequestHandler`` stack.
"""

from __future__ import annotations

import os
from http import HTTPStatus
from typing import Protocol

from polylogue.daemon.process_start import started_at_wall, uptime_seconds
from polylogue.logging import get_logger

logger = get_logger(__name__)


class ProbeResponder(Protocol):
    """Subset of ``DaemonAPIHandler`` the probe handlers depend on."""

    def _send_json(self, status: HTTPStatus, payload: object) -> None: ...


def handle_healthz_live(responder: ProbeResponder) -> None:
    """Liveness probe — `GET /healthz/live`.

    Kubernetes liveness contract: "is this process alive enough to
    answer?" The HTTP handler returning at all is the answer; the body
    is a small structured payload for human/operator inspection.

    This endpoint MUST NOT touch the database, filesystem (beyond what
    Python needs to serialize JSON), or any subsystem that could block.
    A liveness probe blocking on a stuck subsystem causes k8s to kill
    and restart the pod — which is precisely the failure mode liveness
    is supposed to detect, but only when the *process* itself is wedged.
    """
    from polylogue.daemon.lifecycle import process_heartbeat_age_seconds

    uptime_s = uptime_seconds()
    heartbeat_age_s = process_heartbeat_age_seconds()
    responder._send_json(
        HTTPStatus.OK,
        {
            "status": "alive",
            "pid": os.getpid(),
            "started_at": started_at_wall(),
            "uptime_s": round(uptime_s, 3),
            "heartbeat_age_s": None if heartbeat_age_s is None else round(heartbeat_age_s, 3),
        },
    )


def handle_healthz_ready(responder: ProbeResponder) -> None:
    """Readiness probe — `GET /healthz/ready`.

    Kubernetes readiness contract: "should traffic be routed to this
    pod?" Returns 200 when the daemon is ready to serve requests and
    503 (with structured ``reason`` from a closed taxonomy) otherwise.

    Readiness signals:

    - degraded process flag (set on schema mismatch, etc.) — hard NOT_READY,
      ``reason`` carries the explicit ``DegradedReason.code``
    - schema version matches runtime — required
    - FAST tier health checks (no CRITICAL alerts) — required

    Bounded to FAST tier (< 1s budget). Never runs MEDIUM/EXPENSIVE
    checks; the probe is called frequently (k8s default: every 10s).

    Closed reason taxonomy:

    - ``schema_version_mismatch`` — schema_version check is CRITICAL
    - ``critical_check_failed`` — some other FAST check is CRITICAL
    - ``fts_not_fresh`` — archive FTS invariant is not ready
    - ``probe_error`` — internal failure executing the probe itself
    - any ``DegradedReason.code`` — process-local degraded flag is set
    """
    try:
        from polylogue.core.degraded import degraded_reason
        from polylogue.daemon.health import HealthSeverity, HealthTier, check_health

        reason = degraded_reason()
        if reason is not None:
            detail = dict(reason.detail) if reason.detail is not None else None
            responder._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "status": "not_ready",
                    "reason": reason.code,
                    "message": reason.message,
                    "detail": detail,
                },
            )
            return

        health = check_health(tiers={HealthTier.FAST})
        checks = [
            {
                "name": alert.check_name,
                "severity": alert.severity.value,
                "message": alert.message,
            }
            for alert in health.alerts
        ]
        schema_alert = next(
            (a for a in health.alerts if a.check_name == "schema_version"),
            None,
        )
        schema_ok = schema_alert is None or schema_alert.severity == HealthSeverity.OK
        critical = [a for a in health.alerts if a.severity == HealthSeverity.CRITICAL]
        from polylogue.daemon.fts_status import fts_readiness_info
        from polylogue.paths import active_index_db_path

        dbf = active_index_db_path()
        fts_ready = True
        fts_payload: dict[str, object] | None = None
        if dbf.exists():
            fts_payload = fts_readiness_info(dbf)
            fts_ready = bool(fts_payload.get("invariant_ready", False))

        if schema_ok and not critical and fts_ready:
            responder._send_json(
                HTTPStatus.OK,
                {
                    "status": "ready",
                    "overall": health.overall_status.value,
                    "checks": checks,
                    "fts": fts_payload,
                },
            )
            return

        if not schema_ok:
            reason_code = "schema_version_mismatch"
        elif critical:
            reason_code = "critical_check_failed"
        else:
            reason_code = "fts_not_fresh"
        responder._send_json(
            HTTPStatus.SERVICE_UNAVAILABLE,
            {
                "status": "not_ready",
                "reason": reason_code,
                "overall": health.overall_status.value,
                "checks": checks,
                "fts": fts_payload,
            },
        )
    except Exception as exc:
        # Probe must always answer with a structured 503 rather than
        # leak as a 500 — we genuinely want broad Exception here.
        logger.exception("readiness probe failed")
        responder._send_json(
            HTTPStatus.SERVICE_UNAVAILABLE,
            {
                "status": "not_ready",
                "reason": "probe_error",
                "message": str(exc),
            },
        )


__all__ = [
    "handle_healthz_live",
    "handle_healthz_ready",
]
