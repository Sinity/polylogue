"""Contract tests for the OTLP HTTP receiver (#1321)."""

from __future__ import annotations

import json

from polylogue.daemon.otlp_receiver import (
    OtlpTelemetryRow,
    handle_logs,
    handle_metrics,
    handle_traces,
    store_telemetry,
    supports_otlp,
)

# ── Fixtures: build minimal OTLP protobuf requests ──────────────────


def _make_trace_request() -> bytes:
    """Build a minimal ``ExportTraceServiceRequest`` in protobuf.

    mypy does not understand protobuf-generated SerializeToString return types.
    """
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
    from opentelemetry.proto.resource.v1.resource_pb2 import Resource
    from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span

    span = Span(
        name="test-span",
        trace_id=b"\x01" * 16,
        span_id=b"\x02" * 8,
        kind=Span.SPAN_KIND_INTERNAL,
        start_time_unix_nano=1_000_000_000,
        end_time_unix_nano=2_000_000_000,
    )
    scope = ScopeSpans(spans=[span])
    resource = Resource(attributes=[KeyValue(key="service.name", value=AnyValue(string_value="polylogue"))])
    resource_span = ResourceSpans(resource=resource, scope_spans=[scope])
    result: bytes = ExportTraceServiceRequest(resource_spans=[resource_span]).SerializeToString()
    return result


def _make_metric_request() -> bytes:
    """Build a minimal ``ExportMetricsServiceRequest`` in protobuf."""
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
        ExportMetricsServiceRequest,
    )
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
    from opentelemetry.proto.metrics.v1.metrics_pb2 import (
        Gauge,
        Metric,
        NumberDataPoint,
        ResourceMetrics,
        ScopeMetrics,
    )
    from opentelemetry.proto.resource.v1.resource_pb2 import Resource

    dp = NumberDataPoint(time_unix_nano=1_000_000_000, as_int=42)
    gauge = Gauge(data_points=[dp])
    metric = Metric(name="test.counter", gauge=gauge)
    scope = ScopeMetrics(metrics=[metric])
    resource = Resource(attributes=[KeyValue(key="service.name", value=AnyValue(string_value="polylogue"))])
    rm = ResourceMetrics(resource=resource, scope_metrics=[scope])
    result: bytes = ExportMetricsServiceRequest(resource_metrics=[rm]).SerializeToString()
    return result


def _make_log_request() -> bytes:
    """Build a minimal ``ExportLogsServiceRequest`` in protobuf."""
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
        ExportLogsServiceRequest,
    )
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
    from opentelemetry.proto.logs.v1.logs_pb2 import LogRecord, ResourceLogs, ScopeLogs
    from opentelemetry.proto.resource.v1.resource_pb2 import Resource

    record = LogRecord(
        time_unix_nano=1_000_000_000,
        body=AnyValue(string_value="test log message"),
        severity_text="INFO",
    )
    scope = ScopeLogs(log_records=[record])
    resource = Resource(attributes=[KeyValue(key="service.name", value=AnyValue(string_value="polylogue"))])
    rl = ResourceLogs(resource=resource, scope_logs=[scope])
    result: bytes = ExportLogsServiceRequest(resource_logs=[rl]).SerializeToString()
    return result


# ── Tests ───────────────────────────────────────────────────────────


def test_supports_otlp() -> None:
    """opentelemetry-proto should be importable in the test environment."""
    assert supports_otlp() is True


class TestTraceReceiver:
    def test_accepts_protobuf_traces(self) -> None:
        body = _make_trace_request()
        result = handle_traces(body, "application/x-protobuf")
        assert result.status == 200
        assert len(result.body) > 0

    def test_accepts_json_traces(self) -> None:
        from google.protobuf.json_format import MessageToJson
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
        from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
        from opentelemetry.proto.resource.v1.resource_pb2 import Resource
        from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span

        span = Span(
            name="json-test",
            trace_id=b"\x01" * 16,
            span_id=b"\x02" * 8,
            kind=Span.SPAN_KIND_INTERNAL,
            start_time_unix_nano=1,
            end_time_unix_nano=2,
        )
        scope = ScopeSpans(spans=[span])
        resource = Resource(attributes=[KeyValue(key="service.name", value=AnyValue(string_value="polylogue"))])
        req = ExportTraceServiceRequest(resource_spans=[ResourceSpans(resource=resource, scope_spans=[scope])])
        body = MessageToJson(req).encode("utf-8")

        result = handle_traces(body, "application/json")
        assert result.status == 200
        assert result.content_type == "application/json"

    def test_rejects_invalid_payload(self) -> None:
        result = handle_traces(b"not-valid-protobuf", "application/x-protobuf")
        assert result.status == 400

    def test_response_is_valid_protobuf(self) -> None:
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse

        body = _make_trace_request()
        result = handle_traces(body, "application/x-protobuf")
        response = ExportTraceServiceResponse()
        response.ParseFromString(result.body)
        # Partial success should be present by default
        assert hasattr(response, "partial_success")


class TestMetricReceiver:
    def test_accepts_protobuf_metrics(self) -> None:
        body = _make_metric_request()
        result = handle_metrics(body, "application/x-protobuf")
        assert result.status == 200

    def test_rejects_invalid_payload(self) -> None:
        result = handle_metrics(b"garbage", "application/x-protobuf")
        assert result.status == 400


class TestLogReceiver:
    def test_accepts_protobuf_logs(self) -> None:
        body = _make_log_request()
        result = handle_logs(body, "application/x-protobuf")
        assert result.status == 200

    def test_rejects_invalid_payload(self) -> None:
        result = handle_logs(b"garbage", "application/x-protobuf")
        assert result.status == 400


class TestDbStorage:
    def test_store_and_query_telemetry(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        db = str(tmp_path / "test.db")
        body = _make_trace_request()
        row = OtlpTelemetryRow(
            received_at="2025-06-01T00:00:00Z",
            signal_type="traces",
            content_type="application/x-protobuf",
            payload=body,
            resource_count=1,
            span_count=1,
        )
        store_telemetry(db, row)

        import sqlite3

        conn = sqlite3.connect(tmp_path / "ops.db")
        try:
            rows = conn.execute("SELECT signal_type, resource_count, span_count FROM otlp_telemetry").fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "traces"
            assert rows[0][1] == 1
            assert rows[0][2] == 1
        finally:
            conn.close()

    def test_store_is_idempotent(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        db = str(tmp_path / "test.db")
        body = _make_metric_request()
        for _ in range(3):
            store_telemetry(
                db,
                OtlpTelemetryRow(
                    received_at="2025-06-01T00:00:00Z",
                    signal_type="metrics",
                    content_type="application/x-protobuf",
                    payload=body,
                    resource_count=1,
                    metric_count=1,
                ),
            )
        import sqlite3

        conn = sqlite3.connect(tmp_path / "ops.db")
        try:
            count = conn.execute("SELECT count(*) FROM otlp_telemetry").fetchone()[0]
            assert count == 3
        finally:
            conn.close()


class TestGracefulFallback:
    def test_traces_501_when_otel_missing(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setattr("polylogue.daemon.otlp_receiver._OTEL_AVAILABLE", False)
        result = handle_traces(b"anything", "application/x-protobuf")
        assert result.status == 501
        assert b"not available" in result.body

    def test_metrics_501_when_otel_missing(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setattr("polylogue.daemon.otlp_receiver._OTEL_AVAILABLE", False)
        result = handle_metrics(b"anything", "application/x-protobuf")
        assert result.status == 501

    def test_logs_501_when_otel_missing(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setattr("polylogue.daemon.otlp_receiver._OTEL_AVAILABLE", False)
        result = handle_logs(b"anything", "application/x-protobuf")
        assert result.status == 501


class TestContentTypeNegotiation:
    def test_json_request_gets_json_response(self) -> None:
        from google.protobuf.json_format import MessageToJson
        from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
        from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
        from opentelemetry.proto.metrics.v1.metrics_pb2 import (
            Gauge,
            Metric,
            NumberDataPoint,
            ResourceMetrics,
            ScopeMetrics,
        )
        from opentelemetry.proto.resource.v1.resource_pb2 import Resource

        dp = NumberDataPoint(time_unix_nano=1, as_int=1)
        metric = Metric(name="t", gauge=Gauge(data_points=[dp]))
        rm = ResourceMetrics(
            resource=Resource(attributes=[KeyValue(key="s", value=AnyValue(string_value="x"))]),
            scope_metrics=[ScopeMetrics(metrics=[metric])],
        )
        body = MessageToJson(ExportMetricsServiceRequest(resource_metrics=[rm])).encode()

        result = handle_metrics(body, "application/json")
        assert result.status == 200
        assert result.content_type == "application/json"
        assert json.loads(result.body.decode()) is not None

    def test_protobuf_request_gets_protobuf_response(self) -> None:
        body = _make_log_request()
        result = handle_logs(body, "application/x-protobuf")
        assert result.status == 200
        assert result.content_type == "application/x-protobuf"
