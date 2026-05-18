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
import time
from http import HTTPStatus
from typing import Protocol

from polylogue.logging import get_logger

logger = get_logger(__name__)


# Process start time captured at module import — used by /healthz/live to
# report uptime without touching disk, DB, or any other subsystem.
_PROCESS_STARTED_AT: float = time.monotonic()
_PROCESS_STARTED_AT_WALL: float = time.time()


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
    uptime_s = time.monotonic() - _PROCESS_STARTED_AT
    responder._send_json(
        HTTPStatus.OK,
        {
            "status": "alive",
            "pid": os.getpid(),
            "started_at": _PROCESS_STARTED_AT_WALL,
            "uptime_s": round(uptime_s, 3),
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

    - ``schema_incompatible`` — schema_version check is CRITICAL
    - ``critical_check_failed`` — some other FAST check is CRITICAL
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
        if schema_ok and not critical:
            responder._send_json(
                HTTPStatus.OK,
                {
                    "status": "ready",
                    "overall": health.overall_status.value,
                    "checks": checks,
                },
            )
            return

        reason_code = "schema_incompatible" if not schema_ok else "critical_check_failed"
        responder._send_json(
            HTTPStatus.SERVICE_UNAVAILABLE,
            {
                "status": "not_ready",
                "reason": reason_code,
                "overall": health.overall_status.value,
                "checks": checks,
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
