"""HTTP adaptation for the shared Prometheus metric product."""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from typing import Protocol

from polylogue.operations import daemon_metrics

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class MetricsResponder(Protocol):
    """Text response boundary supplied by the daemon HTTP server."""

    def _send_text(self, status: HTTPStatus, body: str, *, content_type: str) -> None: ...


def handle_metrics(responder: MetricsResponder, db: Path) -> None:
    """Publish one product-owned scrape using the existing HTTP contract."""
    responder._send_text(HTTPStatus.OK, daemon_metrics.render_metrics(db), content_type=PROMETHEUS_CONTENT_TYPE)


__all__ = ["PROMETHEUS_CONTENT_TYPE", "MetricsResponder", "handle_metrics"]
