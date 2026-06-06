"""OTLP HTTP receiver — accepts traces, metrics, and logs via OTLP/HTTP (#1321).

Implements ``POST /v1/traces``, ``POST /v1/metrics``, ``POST /v1/logs``
with protobuf and JSON content-type support.  Compatible with Jaeger,
Tempo, Grafana Agent, and any OpenTelemetry SDK that exports via
``otlp/http``.

Design notes:

- **Opt-in only** — the routes are gated on the
  ``observability_enabled`` config flag (default off, env override
  ``POLYLOGUE_OBSERVABILITY_ENABLED``). When the flag is off the
  daemon returns ``404 Not Found`` so the receiver doesn't advertise
  itself. Closes #1604, which documented that an earlier version of
  this docstring claimed gating that was not actually implemented.
- **Loopback unauthenticated; remote requires the daemon auth token**
  — when ``observability_enabled`` is on AND the daemon is bound
  non-loopback, the route runs through ``_check_auth`` so exporters
  that DO carry credentials are the only safe non-loopback case.
  Loopback callers still bypass auth, matching ``/metrics`` and
  ``/healthz/*``.
- **Body cap** — ``Content-Length`` is capped at
  ``otlp_max_body_bytes`` (default 8 MiB; override
  ``POLYLOGUE_OTLP_MAX_BODY_BYTES``); over-limit POSTs receive
  ``413 Payload Too Large``.
- **Graceful fallback** — when ``opentelemetry-proto`` is not installed
  the endpoints return ``501 Not Implemented`` rather than crashing.
- **Content-type negotiation** — ``application/x-protobuf`` and
  ``application/json`` are both accepted.
- **Partial success** — every response includes a ``partial_success``
  block so exporters can track which spans/metrics/logs were rejected.
- **Storage** — received telemetry is persisted in ``otlp_telemetry``
  in archive ``ops.db`` with signal-type, resource/span/metric/log counts,
  and the raw payload.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

from polylogue.logging import get_logger
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

logger = get_logger(__name__)

# ── Graceful import of opentelemetry-proto ──────────────────────────

_OTEL_AVAILABLE = False
_ExportTraceServiceRequest: Any = None
_ExportTraceServiceResponse: Any = None
_ExportMetricsServiceRequest: Any = None
_ExportMetricsServiceResponse: Any = None
_ExportLogsServiceRequest: Any = None
_ExportLogsServiceResponse: Any = None

try:
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
        ExportLogsServiceRequest,
        ExportLogsServiceResponse,
    )
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
        ExportMetricsServiceRequest,
        ExportMetricsServiceResponse,
    )
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
        ExportTraceServiceResponse,
    )

    _OTEL_AVAILABLE = True
except ImportError:
    logger.info("otlp: opentelemetry-proto not installed — OTLP endpoints return 501")


# ── Public API ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OtlpResult:
    status: int  # HTTP status code
    body: bytes  # Response body (protobuf or JSON)
    content_type: str  # Response content-type


@dataclass(frozen=True, slots=True)
class OtlpTelemetryRow:
    received_at: str
    signal_type: str
    content_type: str
    payload: bytes
    resource_count: int
    span_count: int | None = None
    metric_count: int | None = None
    log_record_count: int | None = None


def handle_traces(body: bytes, content_type: str, db_path: str | None = None) -> OtlpResult:
    """Accept an OTLP ``ExportTraceServiceRequest`` and return the response."""
    if not _OTEL_AVAILABLE:
        return _not_implemented("traces")
    return _handle_signal(body, content_type, "traces", ExportTraceServiceRequest, ExportTraceServiceResponse, db_path)


def handle_metrics(body: bytes, content_type: str, db_path: str | None = None) -> OtlpResult:
    """Accept an OTLP ``ExportMetricsServiceRequest`` and return the response."""
    if not _OTEL_AVAILABLE:
        return _not_implemented("metrics")
    return _handle_signal(
        body, content_type, "metrics", ExportMetricsServiceRequest, ExportMetricsServiceResponse, db_path
    )


def handle_logs(body: bytes, content_type: str, db_path: str | None = None) -> OtlpResult:
    """Accept an OTLP ``ExportLogsServiceRequest`` and return the response."""
    if not _OTEL_AVAILABLE:
        return _not_implemented("logs")
    return _handle_signal(body, content_type, "logs", ExportLogsServiceRequest, ExportLogsServiceResponse, db_path)


def supports_otlp() -> bool:
    """Return True if the OTLP receiver can accept requests."""
    return _OTEL_AVAILABLE


def store_telemetry(db_path: str, row: OtlpTelemetryRow) -> None:
    """Persist received telemetry to the archive ops database."""
    try:
        ops_db = _ops_db_path(db_path)
        initialize_archive_database(ops_db, ArchiveTier.OPS)
        conn = sqlite3.connect(ops_db)
        try:
            conn.execute(
                "INSERT INTO otlp_telemetry (received_at_ms, signal_type, content_type, payload, resource_count, span_count, metric_count, log_record_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _epoch_ms(row.received_at),
                    row.signal_type,
                    row.content_type,
                    row.payload,
                    row.resource_count,
                    row.span_count,
                    row.metric_count,
                    row.log_record_count,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("otlp: failed to store telemetry: %s", exc)


def _ops_db_path(db_path: str) -> Path:
    path = Path(db_path)
    return path if path.name == "ops.db" else path.with_name("ops.db")


def _epoch_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


# ── Internal helpers ────────────────────────────────────────────────


def _not_implemented(signal: str) -> OtlpResult:
    return OtlpResult(
        status=501,
        body=json.dumps(
            {"error": f"OTLP/{signal} not available — install opentelemetry-proto", "signal": signal}
        ).encode(),
        content_type="application/json",
    )


def _handle_signal(
    body: bytes,
    content_type: str,
    signal_type: str,
    request_cls: Any,
    response_cls: Any,
    db_path: str | None,
) -> OtlpResult:
    """Parse *body* as an OTLP request, persist, and return the response."""
    request = _parse_request(body, content_type, request_cls)
    if request is None:
        return OtlpResult(status=400, body=b'{"error":"invalid request payload"}', content_type="application/json")

    # Count resources and signal-specific entities for storage metadata.
    resource_count, entity_count = _count_entities(request, signal_type)

    # Build a response with partial success (field name varies by signal type).
    response = response_cls()
    rejected = 0
    if hasattr(response, "partial_success"):
        if signal_type == "metrics":
            response.partial_success.rejected_data_points = rejected
        elif signal_type == "logs":
            response.partial_success.rejected_log_records = rejected
        else:
            response.partial_success.rejected_spans = rejected

    # Serialise the response in the same format as the request.
    response_body = _serialize_response(response, content_type)

    # Persist the raw request for audit/query.
    received_at = datetime.now(tz=timezone.utc).isoformat()
    if db_path:
        try:
            store_telemetry(
                db_path,
                OtlpTelemetryRow(
                    received_at=received_at,
                    signal_type=signal_type,
                    content_type=content_type,
                    payload=body,
                    resource_count=resource_count,
                    span_count=entity_count if signal_type == "traces" else None,
                    metric_count=entity_count if signal_type == "metrics" else None,
                    log_record_count=entity_count if signal_type == "logs" else None,
                ),
            )
        except Exception:
            logger.debug("otlp: db persist skipped (no db_path or write failed)")

    if content_type.startswith("application/json"):
        return OtlpResult(status=200, body=response_body, content_type="application/json")
    return OtlpResult(status=200, body=response_body, content_type="application/x-protobuf")


def _parse_request(body: bytes, content_type: str, request_cls: Any) -> Any | None:
    """Parse an OTLP request from protobuf or JSON."""
    try:
        if "json" in content_type:
            from google.protobuf.json_format import Parse as ProtoParse

            return ProtoParse(body.decode("utf-8"), request_cls())
        else:
            request = request_cls()
            request.ParseFromString(body)
            return request
    except Exception as exc:
        logger.warning("otlp: failed to parse %s request: %s", request_cls.__name__, exc)
        return None


def _serialize_response(response: Any, content_type: str) -> bytes:
    """Serialise an OTLP response to protobuf or JSON."""
    if "json" in content_type:
        from google.protobuf.json_format import MessageToJson

        return bytes(MessageToJson(response).encode("utf-8"))
    else:
        return bytes(response.SerializeToString())


def _count_entities(request: Any, signal_type: str) -> tuple[int, int]:
    """Return (resource_count, entity_count) for an OTLP request."""
    resources = 0
    entities = 0
    try:
        if signal_type == "traces" and hasattr(request, "resource_spans"):
            for rs in request.resource_spans:
                resources += 1
                for ss in rs.scope_spans:
                    entities += len(ss.spans)
        elif signal_type == "metrics" and hasattr(request, "resource_metrics"):
            for rm in request.resource_metrics:
                resources += 1
                for sm in rm.scope_metrics:
                    entities += len(sm.metrics)
        elif signal_type == "logs" and hasattr(request, "resource_logs"):
            for rl in request.resource_logs:
                resources += 1
                for sl in rl.scope_logs:
                    entities += len(sl.log_records)
    except Exception:
        pass
    return resources, entities
